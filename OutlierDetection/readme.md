# Collection of outlier detection methods: 


## Ideas and tools: 

- Numeric methods: IQR (InterQuartile Range), Z-Score, mahalanobis distance

- proximity based models: kNN

- linear models: PCA (the reconstruction error for outliers are larger), One-class SVM 

- Autoencoder

- PyOD

- Isolation Forest: Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

- DBSCAN clustering: density based outlier detection method in a one or multi dimensional feature space

- 看做数据不平衡下的分类问题，通过有监督学习分类

- semi-supervised learning: Anomaly Detection with Partially Observed Anomalies (https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/www18bw.pdf)


## References: 

https://pyod.readthedocs.io/en/latest/

https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/

https://scikit-learn.org/stable/modules/outlier_detection.html

https://scikit-learn.org/stable/auto_examples/plot_anomaly_comparison.html#sphx-glr-auto-examples-plot-anomaly-comparison-py

https://www.kdnuggets.com/2018/12/four-techniques-outlier-detection.html

https://github.com/josephmisiti/awesome-machine-learning#python-general-purpose

https://github.com/rob-med/awesome-TS-anomaly-detection

https://www.zhihu.com/question/280696035

https://zhuanlan.zhihu.com/p/58313521
