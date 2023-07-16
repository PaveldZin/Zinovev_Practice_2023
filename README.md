# Zinovev_Practice_2023
В рамках практики была изучена статья TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering (https://arxiv.org/abs/1812.09551)

В результате удалось имплементировать вариант NoLE (TaxoGen без local embedding). Полный вариант не удалось сделать из-за проблем с пониманием устройства локального эмбеддинга. Их устройство описано в одной и цитируемых статей: Expert Finding in Heterogeneous Bibliographic Networks with Locally-trained Embeddings (https://arxiv.org/abs/1803.03370).

Использованный датасет arxiv: https://www.kaggle.com/datasets/Cornell-University/arxiv

В качестве узла дерева таксономии используется класс Topic. Функция adaptive_clustering выполняет разделение узла на подтемы, и возвращает обновленную тему и список подтем.

В качестве ключевых терминов использовались частые существительные и их связки, но очевидно, что для построения таксономии необходим более тщательный отбор терминов.

Помимо этого выяснилось, что в данный момент нет хорошей имплементации spherical k-means в Python, поэтому использовался k-means с предварительной нормализацией векторов эмбеддинга.
