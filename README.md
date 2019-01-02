## Confidence-based Graph Convolutional Networks for Semi-Supervised Learning

Source code for [AISTATS 2019](https://www.aistats.org/) paper: [Confidence-based Graph Convolutional Networks for Semi-Supervised Learning]().

![](/Users/shikharvashishth/Documents/ConfGCN/overview.png)*Label prediction on node a by Kipf-GCN and ConfGCN (this paper). L0 is a’s true label. Shade
intensity of a node reflects the estimated score of label L1 assigned to that node. Since Kipf-GCN is not capable
of estimating influence of one node on another, it is misled by the dominant label L1 in node a’s neighborhood
and thereby making the wrong assignment. ConfGCN, on the other hand, estimates confidences (shown by bars)
over the label scores, and uses them to increase influence of nodes b and c to estimate the right label on a. Please refer to paper for more details.* 

### Dependencies

- Compatible with TensorFlow 1.x and Python 3.x.
- Dependencies can be installed using `requirements.txt`.

### Dataset:

- We use citation network datasets: Cora, Citeseer, Pubmed, and CoraML for evaluation in our paper.
- Cora, Citeseer, and Pubmed datasets was taken directly from [here](https://github.com/tkipf/gcn/tree/master/gcn/data). CoraML dataset was taken from [here](https://github.com/abojchevski/graph2gauss) and was placed in the same format as other datasets for semi-supervised settings. 
- `data.zip` containing all the datasets in required format can be downloaded from here. 

### Evaluate pretrained model:

- `confgcn.py` contains TensorFlow (1.x) based implementation of **ConfGCN** (proposed method).
- Download the pretrained model's parameters from [RiedelNYT](https://drive.google.com/file/d/1CUk10FTncaaZspAoh8YkHTML3RJHfW7e/view?usp=sharing) and [GIDS](https://drive.google.com/file/d/1X5pKkL6eOkGXw39baq0n9noBXa--5EhE/view?usp=sharing) (put downloaded folders in `checkpoint` directory). 
- Execute `evaluate.sh` for comparing pretrained RESIDE model against baselines (plots Precision-Recall curve). 

### Training from scratch:

- Execute `setup.sh` for downloading GloVe embeddings.

- For training **ConfGCN** run:

  ```shell
  python reside.py -data data/riedel_processed.pkl -name new_run
  ```

### Citation

```tex
@InProceedings{ConfGCN2019,
  author = 	"Vashishth, Shikhar
		and Yadav, Prateek
		and Bhandari, Manik
		and Talukdar, Partha",
  title = 	"Confidence-based Graph Convolutional Networks for Semi-Supervised Learning",
  booktitle = 	"International Conference on  Artificial Intelligence and Statistics (AISTATS)",
  year = 	"2019",
  location = 	"Naha, Okinawa, Japan"
}
```

For any clarification, comments, or suggestions please create an issue or contact [shikhar@iisc.ac.in](http://shikhar-vashishth.github.io).