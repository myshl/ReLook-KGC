"""Detach the full suite after successful preflight; preserve a durable PID/log."""
import json
import os
from pathlib import Path
import subprocess
from paper_suite import ROOT, PYTHON, lock_sources, write_json


def main():
    output=ROOT/'experiments/final-v1'
    output.mkdir(parents=True,exist_ok=True)
    launch=output/'launcher.json'
    if launch.exists():
        raise SystemExit('Already launched. Inspect the saved PID; do not duplicate the suite.')
    smoke=ROOT/'experiments/smoke-v1'
    audit=json.loads((smoke/'smoke_audit.json').read_text())
    assert audit['official_forward_equivalence']['max_abs_logit_difference'] < 1e-6
    assert json.loads((smoke/'DB15K/analysis/replay_verification.json').read_text())['exact_match']
    assert json.loads((smoke/'DB15K/analysis/complete.json').read_text())['status'] in (
        'complete','accuracy_complete_efficiency_pending')
    for dataset in ['DB15K','YAGO15K','FB15k-237']:
        assert (smoke/dataset/'all/complete.json').is_file()
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
               LD_LIBRARY_PATH=str(ROOT/'.cuda_lib')+':'+env.get('LD_LIBRARY_PATH',''))
    with (output/'preflight_tests.log').open('w') as log:
        subprocess.run([str(PYTHON),'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    lock_sources(output)
    command=[str(PYTHON),str(ROOT/'script/paper_suite.py'),'--output',str(output)]
    with (output/'queue.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL,start_new_session=True)
    write_json(launch,dict(status='launched',pid=process.pid,command=command))
    print(json.dumps(dict(pid=process.pid,output=str(output))))


if __name__=='__main__':
    main()
