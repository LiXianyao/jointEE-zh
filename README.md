主要针对ACE-2005事件抽取的端到端方法。
是一个堆档整理repo，之前有个做在English数据集上的，版本控制没做好，太杂，这次重构一下

数据集的处理可使用：
```angular2html
https://github.com/LiXianyao/ace2005-preprocessing-With-Chinese-Branch-/tree/chinese
```

处理完的数据集情况（可能）如下
|          | Documents    |  Sentences   |Triggers    | Arguments | Entity Mentions  |
|-------   |--------------|--------------|------------|-----------|----------------- |
| Test     | 64        | 557           | 305           | 796             |  3436             |
| Dev      | 20        | 271           | 117           |254              |  1376             |
| Train    | 549       | 5801         | 2826          | 6892             |   33680            |

涉及到是否采用标题句子、以哪些符号断句、所标注的事件/触发词是否在同个断句内等策略，数字可能有一定浮动。ACE数据集上的划分只有按文件为单位，未见具体的sentence level处理

目前，使用albert_chinese_tiny的baseline表现为
|          | P    |  R   |F1    | 
|-------   |--------------|--------------|------------|
| Trigger Classification     | 66.2        | 67.14           | 66.67           |          
| Entity Mention Detection      | 80.86        | 80.44           | 80.65          |          
| Argument Role Prediction    | 35.57   |    38.19     |     36.83      |          

