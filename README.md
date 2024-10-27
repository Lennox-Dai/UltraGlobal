# UltraGlobal

> This is the repository of paper : "UltraGlobal: An Enhanced Approach to Image Retrieval Using Global Features"

### Install

- You may refer to our requirement to set up the same environment with our original code.

  *requirements.txt*

### Set up

- You can refer to [ShihaoShao-GH/SuperGlobal: ICCV 2023 Paper Global Features are All You Need for Image Retrieval and Reranking Official Repository](https://github.com/ShihaoShao-GH/SuperGlobal) to set up the respository.

### Evaluation

- You can run the instruction below to evaluate the model:

  ```
   python test.py MODEL.DEPTH 50 TEST.WEIGHTS ./weights TEST.DATA_DIR ./revisitop SupG.gemp True SupG.rgem True SupG.sgem True SupG.relup True SupG.rerank True SupG.onemeval False
  ```

### Toxic finetuning

- You should run *new_try.py* under folder *finetune* to do toxic finetuning.

- The model is saved under *./finetune/model*  run the following to evaluate the toxic model:

  ```
  python test.py MODEL.DEPTH 50 TEST.WEIGHTS ./finetune/model TEST.DATA_DIR ./revisitop SupG.gemp True SupG.rgem True SupG.sgem True SupG.relup True SupG.rerank True SupG.onemeval False
  ```

  