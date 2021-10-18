
比赛&&数据集地址[https://challenge.xfyun.cn/topic/info?type=user-portrait&ch=dc-web-20](https://challenge.xfyun.cn/topic/info?type=user-portrait&ch=dc-web-20)

思路:

将用户的基本特征与行为序列特征分开考虑，在这里基本特征由于比较少，我直接使用了最简单的one-hot，行为特征的构建首先要经过一层词向量模型的训练，这里可以用word2vec,fasttext(bert感觉用不上，这里的词表都是不规则的数字脱敏后处理的)，接下来可以使用GRU,RCNN,Transformer这类去做，最好的是GRU+Attention,这个能上0.7+,排名也能上10%,前排大佬可能有更好的比如对抗训练或者数据增强的方法，模型没有融合，可以再一步优化

![image](https://user-images.githubusercontent.com/41276134/137570452-9bcee0bb-eeed-471a-afb8-db0788ad5f96.png)

