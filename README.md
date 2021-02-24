# Probing-Vokenization

## Basics

Server: nlp7.cs.ucla.edu

Root PATH: /local/enhungchu/Probing-Vokenization/

Dataset PATH: ./contextual-repr-analysis/data/

## Running Tasks

- Generate hdf5 file for the model

  - BERT-Paper: 
    ```cd contextual-repr-analysis```

    ```bash
    ./scripts/precontextualize_bert_base_cased.sh \
    ../bert-nelson/extract_features.py contextualizers/bert_base_cased \
    ../cased_L-12_H-768_A-12/bert_model.ckpt.index ../cased_L-12_H-768_A-12/bert_config.json ../cased_L-12_H-768_A-12/vocab.txt
    ```

  - Vokenization, BERT-wiki, BERT-uncased(pytorch models)
    NOTE: This step has some package conflicts, create a separate conda env for this step.
    ```cd vokenization```

    ```bash
    python extract_features.py \
    --input_file=../contextual-repr-analysis/data/pos/en_ewt-ud_sentences.txt \
    --output_file=../contextual-repr-analysis/contextualizers/bert_wiki/ewt_pos.hdf5 \
    --bert_model=bert-base-uncased --model_path='./bert_12L_768H_wiki/' --do_lower_case
    ```

    --model_path specifies the path of the model we're using, the default is the vokenization model.

- Train the probing task on a specific model
  ```cd contextual-repr-analysis```

  ```bash
  allennlp train experiment_configs/bert_base_cased/ewt_pos_tagging.json \
  -s models/bert_base_cased/ --include-package contexteval
  ```

  This trains the top most layer of the bert_base_cased model, the model path a dataset path is specified in the .json file. -s specifies the output path.

  To train all layers of a model, using BERT(13 layers) as an example:

  ```bash
  for ((i=0; i<13; i++));
  do
  allennlp train experiment_configs/bert_base_cased/ewt_pos_tagging.json \
  -s models/bert_base_cased/ewt_pos_tagging_layer_${i} \
  --include-package contexteval \
  --overrides '{"dataset_reader": {"contextualizer": {"layer_num": '${i}'}}, "validation_dataset_reader": {"contextualizer": {"layer_num": '${i}'}}}';
  done
  ```

  