ReLook-KGC: Progressive Reasoning with Visual Refinement
=======================================================

This repository preserves the code corresponding to the **pre-rerun paper
version**, identified by the PDF fingerprint in ``release_manifest.json``.
It does not include the later common-protocol baseline reruns or the early
Soft Prompt ensemble.

The model first reasons with structure and text through layer-wise soft
prompts, then applies shared/directional experts, followed by a visual score
correction. In this release, visual refinement acts on candidate scores;
it is **not** the experimental variant that inserts visual prompts into the
upper Transformer layers.

Paper results
-------------

The archived local results are available in ``results/main_results.csv``.
These are recorded results, not experiments rerun during this code release.

.. list-table:: Validation-selected ReLook-KGC, full filtered test
   :header-rows: 1

   * - Dataset
     - MRR
     - Hits@1
     - Hits@3
     - Hits@10
   * - DB15K (historical)
     - .4655
     - .4035
     - .4947
     - .5874
   * - YAGO15K
     - .4422
     - .3629
     - .4865
     - .5864
   * - FB15K-237
     - .3692
     - .2783
     - .4036
     - .5520

DB15K contains 2,488 unique triples shared between training and test. Its
results are retained to identify this paper snapshot, **not as clean held-out
generalization evidence**. Literature datasets with the same name need not
have the same entity vocabulary or splits.

YAGO15K and FB15K-237 use continued training and validation selection beyond
the original local CSProm-KG reference. The separate trained-backbone rows
in the CSV distinguish this gain from subsequent adaptation. The shared and
hierarchical stages retained their incoming checkpoints on these two datasets.

Source organization
-------------------

::

    versions/db15k/      Original frozen snapshot; DB15K and local CSProm controls
    versions/localplus/  Continued-training snapshot; YAGO15K and FB15K-237
    configs/            Archived stage commands, configurations and deployment
    results/            Aggregate local paper results, without dataset records
    scripts/paper_run.py Portable stage/evaluation entry point
    source_checksums.json  SHA-256 of unmodified historical source files
    input_checksums.json   SHA-256 of the external dataset/feature inputs
    release_manifest.json  Paper identity, selected checkpoints and policies

Within each source snapshot, ``models/relook_model.py`` implements the model
and objectives; ``models/relook.py`` contains the experts, gates and visual
scorer; ``models/bert_for_layerwise.py`` implements layer-wise prompting.
``kgc_data.py`` handles the two prediction directions, ``main.py`` trains and
validates, and ``script/final_analysis.py`` provides the full filtered evaluator
and validation calibration. The original orchestration scripts are retained
as historical records and assume their original directory layout. Use the
portable entry point below to supply new paths.

Older identifiers and the original ``experiments/PROTOCOL.md`` remain unchanged
inside the snapshots for provenance. In particular, older code calls the last
component a selective adapter/router; the current paper calls it **Visual
Refinement**. The release manifest and dataset-specific archived commands take
precedence over the old generic protocol for the final paper settings.

Installation
------------

Use Python 3.10 and a standard virtual environment; **do not use Conda**.

.. code-block:: bash

    python3.10 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m nltk.downloader stopwords
    .venv/bin/python scripts/paper_run.py verify

The historical experiments used PyTorch 1.13.1 with CUDA 11.7, Lightning
1.9.5 and Transformers 4.26.1. A compatible NVIDIA driver is needed for GPU
runs. The original training commands were run on 80-GB GPUs; no smaller-GPU
memory guarantee is implied by this release. The training/evaluation wrappers
do not silently change batches or losses to accommodate different hardware.

External assets
---------------

No datasets, image features, pretrained models, checkpoints, paper PDFs or
credentials are distributed here. Obtain these separately from an authorized
source, preserving entity/relationship ID order. The expected layout is:

::

    assets/
      data/processed/{DB15K,YAGO15K,FB15k-237}/
        entity2id.txt
        relation2id.txt
        entityid2name.txt
        relationid2name.txt
        entityid2description.txt
        train2id.txt
        valid2id.txt
        test2id.txt
      features/
        <dataset>_img_features.pkl
        <dataset>_entity_embed.pkl
        <dataset>_relation_embed.pkl
      model_path/bert-base-uncased/
        ... Hugging Face model and tokenizer files ...

Feature filenames use ``FB15K-237`` (capital K), while the dataset directory
and CLI use ``FB15k-237``. Triple files begin with a count and use the order
``head tail relation``. Text mapping files have a header and tab-separated
ID/text rows. The structural initializer is included as
``versions/db15k/script/pretrain_structure.py``; run it with ``--help`` for
its explicit input/output options. Historical initialization used 50 epochs,
256 dimensions and seed 41504.

Validate external inputs against the paper fingerprints:

.. code-block:: bash

    .venv/bin/python scripts/paper_run.py verify --assets /path/to/assets --dataset YAGO15K

Matching a dataset name alone is insufficient for reproducing a paper number.
Only load trusted pickle/checkpoint files: the historical loader uses Python
pickle and ``torch.load``.

Training
--------

``stage`` replays an archived command with only paths, GPU and output directory
replaced. It prints a dry-run command unless ``--execute`` is supplied. Existing
nonempty output directories are not overwritten.

.. code-block:: bash

    .venv/bin/python scripts/paper_run.py stage --dataset DB15K --stage all \
      --assets /path/to/assets --output outputs/DB15K/all --gpu 0 --execute

    .venv/bin/python scripts/paper_run.py stage --dataset YAGO15K --stage lar_refine \
      --assets /path/to/assets --previous /path/to/local-csprom.ckpt \
      --output outputs/YAGO15K/lar_refine --gpu 0 --execute

For DB15K, the main chain is ``all -> shared -> hierarchical -> visual ->
relation -> query -> prior -> joint``. Supply the validation-selected incoming
checkpoint with ``--previous`` for each later stage. Archived controls include
``csprom``, ``mixed``, ``early6``, ``late6`` and expert controls.

For YAGO15K/FB15K-237, independently train ``ce_refine`` and ``lar_refine``
from the same local CSProm checkpoint; choose the backbone on full validation
MRR, then run the remaining expert/visual/routing stages. Both paper runs
selected ``lar_refine``. Local CSProm initializers were trained with the
``db15k`` source snapshot, despite that directory's short name; their commands
are included in each dataset's ``csprom`` configuration directory.

The archived code contains the validation-only deployment grid and its
tie-breaking rule. Never select checkpoints or policies using test metrics.
The evaluation wrapper below replays the **frozen paper policy**, rather than
performing a fresh search. New training runs require their own validation
selection and must not be presented as identical historical checkpoints.

Full filtered evaluation
------------------------

.. code-block:: bash

    .venv/bin/python scripts/paper_run.py evaluate --dataset YAGO15K \
      --assets /path/to/assets --checkpoint /path/to/selected-visual.ckpt \
      --variant final --gpu 0 --output outputs/YAGO15K/test.json

Use ``--variant backbone`` or ``--variant csprom`` with the corresponding
checkpoint for the local controls. Checkpoint SHA-256 is verified against
the manifest; ``--allow-new-checkpoint`` explicitly permits evaluating a
different checkpoint with the frozen paper policy. Dataset/feature hashes
remain checked. The command evaluates both head and tail queries against all
entities and reports direction-specific and aggregate metrics.

The historical evaluation masks alternative true answers from the union of
training, validation and test triples, retaining the current target. It uses
the original ``torch.argsort`` tie convention, **not** the later average-tie
protocol. Utility supervision separately uses train-filtered reciprocal rank
with strict greater-than comparisons. The training classification loss is not
replaced with a filtered-candidate loss.

Frozen deployment: DB15K blend .5 / threshold 0 (joint stage); YAGO15K blend .5
/ threshold 0 (visual stage); FB15K-237 blend .25 / threshold .1 (prior stage).
All use unbounded evidence-gain caps. DB15K and YAGO15K invoke visual refinement
on all test queries; FB15K-237 invokes it on 87.81%. These are not claims of
uniform selective computation across all datasets.

Tests
-----

.. code-block:: bash

    .venv/bin/python -m unittest discover -s tests -v
    cd versions/db15k
    ../../.venv/bin/python -m unittest discover -s tests -v
    cd ../localplus
    ../../.venv/bin/python -m unittest discover -s tests -v

The release tests and source checks do not retrain models or establish a new
full-test result. Exact numerical reproduction additionally requires the
external assets, selected weights and a compatible runtime.

Attribution
-----------

See ``THIRD_PARTY_NOTICES.txt`` for CSProm-KG and dependency attribution. No
blanket license is asserted over upstream code or external datasets.
