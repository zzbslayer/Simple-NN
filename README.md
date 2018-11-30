# 神经网络作业

## 一、环境与依赖
- Python 3.6
- Numpy
- Pandas
- sklearn
    - sklearn.dataset
        - scikit learn 的数据集，做了可视化，验证神经网络的正确性
    - sklearn.processing
        - 仅仅使用 sklearn.processing.scale 做 z-score 标准化
- matplotlib

## 二、代码说明
- Function: plot_decision_boundary
    - Usage: 画出模型分类结果的分界线
    - Parameter: 
        - X: 数据集的 features
        - y: 数据集的 labels
        - pred_func: 预测函数
    - Example:

        ```
            model = NNmodel(X, y, inputDim=2, outputDim=2)
            model.build_model(3, print_loss=False,activation=activation)
            plot_decision_boundary(X, y, lambda x: model.predict(x))
            plt.title("3 Hidden Layer")
        ```

        ![decision](./image/decision.png)
- 