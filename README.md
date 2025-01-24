# Fine-Tuning IBM Granite LLM for Solving the Heat Equation

This project aims to fine-tune the **IBM Granite Large Language Model (LLM)** to interpret and provide solutions to the **time-dependent heat equation**.

## Heat Equation Overview

The time-dependent heat equation is given as:

$$\frac{\partial T}{\partial t} - \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right) = f(x, y, t)$$


Where:
-  T: Temperature field
-  $$\alpha$$: Thermal conductivity (material property)
- f(x, y, t): Force function representing external influences

---

## Objectives

1. **Fine-tune IBM Granite LLM**: Train the model to interpret the heat equation and its boundary conditions for various scenarios.
2. **Solve Heat Equation**: Generate analytical or numerical solutions based on user-defined inputs for f(x, y, t), $$\alpha \$$, and initial/boundary conditions.

---

## Project Structure
1. The file dataset_creation.ipynb is used for creating the dataset containg question-answer pairs for providing the input to the LLM. </br>
   The dataset can be found at https://www.kaggle.com/datasets/uzumakisumedh/heat-solution-dataset. The dataset is split into train and val pairs during training</br>
2. The file llm_fine-tuning.ipynb contains an implementation of the fine tuning methodology</br>
3. The preprocessing pipeline consists of the following steps
   1. Defining a special pad token to pad the tokens within a batch
   2. Defining a data collator that handles padding efficiently during batch processing
   3. Splitting the dataset into train-val pairs
   4. Tokenizing the train-val datasets using map reduce to be passed to the LLM </br>
4. The evaluation metric that is used both during training and validation is cosine similarity

## Methodology used
The specific methodlogy used for fine-tuning is PEFT(parameter efficient fine tuning). PEFT helps us to train only few of the parameters of the model while freezing the rest. This substantially reduces memory requirements and training time. To read more about PEFT, got to this github-repo: https://github.com/huggingface/peft?tab=readme-ov-file</br>
The files in the heat-solution dataset in  https://www.kaggle.com/datasets/uzumakisumedh/heat-solution-dataset give a detailed overview of the prompts and their various responses.

## Results
Results were analyzed by employing cosine-similarity as the evalaution metric:
```
def compute_metrics(p):
    preds, labels = p

    preds = preds.cpu()
    labels = labels.cpu()
    
    generated_answers = [tokenizer.decode(pred, skip_special_tokens=True) for pred in preds]
    reference_answers = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
    
    cosine_similarities = [compute_cosine_similarity(gen, ref, embedder) for gen, ref in zip(generated_answers, reference_answers)]
    
    avg_cosine_similarity = sum(cosine_similarities) / len(cosine_similarities)
    
    return {'cosine_similarity': avg_cosine_similarity}
```
The model was able to achieve a similarity score of 0.48415 

