# Anchor function: a type of benchmark function for studying language models

主要代码结构如下：
    
    ```
    ├── data_generator 
    ├── model
    ├── utils
    ├── main.py
    ├── data.py
    ├── train.py
    ├── script.py
    ```

运行python script.py即可得到结果，其中script.py可以根据需要进行修改。


文件夹说明：

- data_generator: 用于生成数据集的代码
- model: 模型定义代码
- utils: 作图、设置随机数等工具代码
- result: 保存结果的文件夹
- explore: 用于探索的jupyter notebook
- paper_plot_code: 用于复现论文中的图的jupyter notebook