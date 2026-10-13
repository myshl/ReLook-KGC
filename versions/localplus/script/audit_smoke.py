"""Verify common forward equivalence and actual stage-specific weight changes."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from final_analysis import load_runtime, prepare_batch
from models.csprom_control import CSPromControl
from models.relook_model import ReLookKG
from paper_suite import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    args=p.parse_args()
    torch.set_num_threads(4)
    paths=json.loads((args.suite/'DB15K/selected_checkpoints.json').read_text())
    report={}
    previous=None
    for stage,path in paths.items():
        payload=torch.load(path,map_location='cpu')
        state=payload['state_dict']
        if previous is not None:
            changed=[k for k,v in state.items() if not torch.equal(v,previous[k])]
            report[stage]=dict(changed_tensors=len(changed),changed_names=changed)
            assert changed, f'{stage} made no weight changes'
            prefixes={
                'shared':('shared_deep_prompt_expert.','shared_deep_query_adapter.'),
                'hierarchical':('tail_deep_prompt_expert.','head_deep_prompt_expert.','tail_role_adapter.','head_role_adapter.'),
                'visual':('img_ent_embed.','visual_feature_projection.','visual_rel_embed.','visual_verifier.'),
                'relation':('relation_role_router.',),'query':('relation_role_router.',),
                'prior':('utility_gate.',),'joint':('joint_evidence_router.',),
            }
            assert all(name.startswith(prefixes[stage]) for name in changed), (stage,changed)
        previous=state
    del payload,state,previous
    config,model,dm,gt=load_runtime(paths['all'],args.gpu)
    texts=dict(ent_names=model.ent_names,rel_names=model.rel_names,ent_descs=model.ent_descs)
    official=CSPromControl(config,texts,gt).eval().to(f'cuda:{args.gpu}')
    # Copy learned common weights to prove equations, separate from initialization.
    for target,source in [(official.ent_embed,model.gat_ent_embed),(official.rel_embed,model.gat_rel_embed),
        (official.prompter,model.gat_prompter),(official.fc,model.prompt_fc),
        (official.plm,model.plm),(official.graph_model,model.graph_model)]:
        target.load_state_dict(source.state_dict(),strict=True)
    batch=prepare_batch(next(iter(dm.val_dataloader()[0])),next(model.parameters()).device)
    with torch.inference_mode():
        actual=model._shared_backbone_forward(*batch)[0]
        reference=official(*batch)[0]
    difference=float((actual-reference).abs().max())
    assert difference < 1e-6, difference
    report['official_forward_equivalence']=dict(max_abs_logit_difference=difference,queries=len(batch[0]))
    write_json(args.suite/'smoke_audit.json',report)
    print(json.dumps(report['official_forward_equivalence']))


if __name__=='__main__':
    main()
