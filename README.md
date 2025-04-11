![截图](https://github.com/L11-yy/classify/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-01%20085606.png){width=200}
# 核心实现
class 邮件分类器:
    """支持双特征模式的朴素贝叶斯邮件分类器"""
    
## 多项式朴素贝叶斯
其核心是基于贝叶斯定理与特征条件独立性假设：
### 贝叶斯定理应用：
P(y|x) = P(x|y)P(y)/P(x)
在邮件分类中：
y ∈ {垃圾邮件(1), 普通邮件(0)}
x 表示邮件文本特征向量
通过比较 P(y=1|x) 和 P(y=0|x) 进行分类决策
### 多项式模型特性：
假设特征服从多项式分布
适用于离散型特征（如词频计数）
通过拉普拉斯平滑处理零概率问题

## 数据处理流程
### 文本清洗：
`line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)`  # 移除标点符号和数字
### 中文分词：
`from jieba import cut
line = cut(line)`  # 使用jieba进行中文分词
### 停用词过滤：
`line = filter(lambda word: len(word) > 1, line)`  # 过滤单字词
### 词频统计：
`freq = Counter(chain(*all_docs_words))`  # 统计所有文档的词频

## 特征构建过程
### 高频词特征选择：
特征词 = argmax_{w∈V} count(w), 取前top_n个词
其中V为整个词表，count(w)为词w在所有文档中的出现次数
`self.top_words = [i[0] for i in freq.most_common(self.top_num)]`
### 特点：
简单高效，计算复杂度低
可能包含许多常见但区分度低的词
忽略词在不同类别中的分布差异
### 文档频率特征选择：
特征词 = {w | DF(w) ≥ min_df}
其中DF(w) = |{d∈D | w∈d}|，即包含词w的文档数量
`doc_freq = Counter()
for doc_words in all_docs_words:
    unique_words = set(doc_words)
    doc_freq.update(unique_words)
self.top_words = [word for word, count in doc_freq.items() if count >= self.min_df]`
### 特点：
过滤掉罕见词，提高特征稳定性
保留在多个文档中出现的词
比单纯词频更能反映词的分布特性

## 特征模式切换方法
### 配置参数说明
在EmailClassifier初始化时可通过以下参数控制特征选择模式：
`classifier = EmailClassifier(
    feature_selection='top_words',  # 可选'top_words'或'frequency'
    top_num=100,                   # 高频词模式下的特征词数量
    min_df=5                       # 文档频率模式下的最小文档频率
)`
### 高频词模式
 数学表达
`特征词 = 取词频最高的前N个词`
 使用示例
`分类器 = 邮件分类器(模式='高频词', 特征数=100)`
### TF-IDF模式
 数学表达
`特征词 = {词 | 该词出现的文档数 ≥ 阈值}`
 使用示例
`分类器 = 邮件分类器(模式='文档频率', 最小文档频率=5)`
