from PyQt5.QtWidgets import QMessageBox

def tobedevelop():
    ABOUT = '''
    GraphConstruct: 
    LoadModel: 
    Inference:
    '''
    QMessageBox.information(None, "About", ABOUT)

def guideline():
    GUIDELINE = '''
    <b style="color: red; font-size: 32px;">GUIDELINE</b><br>
    0 介绍<br>
    此APP旨在自动化实现城市功能区识别与可视化。<br>
    具体功能包括：<br>
        POI(语义对象)和地表覆盖(物理对象)实例的图(Graph)构建、可视化、特征统计；<br>
        最小生成树、德劳内三角网以及改进德劳内三角网方法的图构建；<br>
        基于语义对象的UFZ推理、基于物理对象的UFZ推理，以及二者融合的UFZ推理。<br>
    1 数据<br>
    1-1 数据格式<br>
        Shapefile<br>
    1-2 数据列表<br>
        UFZ单元数据要求：属性里需要有哪些内容<br>
        POI数据要求：属性里需要有哪些内容<br>
        地表覆盖数据要求：属性里需要有哪些内容<br>
    2 功能介绍<br>
    2-1 File:用于选择数据来源<br>
    2-2 View:用于可视化被加载的数据<br>
    2-3 GraphConstruct:用于图构建<br>
    2-4 LoadModel:用于加载训练后的模型<br>
    2-5 Inference:用于推理<br>
    '''
    message_box = QMessageBox()
    message_box.setWindowTitle("Guideline")
    message_box.setText(GUIDELINE)
    message_box.exec_()

def about():
    ABOUT = '''
    Author: Yuan Tao 
    Date  : 2024-06-15
    Email : yuantaogiser@gamil.com
    '''
    QMessageBox.information(None, "About", ABOUT)

"""
<p style="text-align: left;">
<b>Guideline</b><br><br>
This is a long message that should automatically wrap to the next line when it reaches the end of the message box width.<br>
You can add extra spaces using &nbsp; like this.&nbsp;&nbsp;&nbsp;See the extra spaces?<br><br>
Here is another line with more text to demonstrate wrapping and spacing.<br>
Remember to use <code>&lt;br&gt;</code> for new lines and <code>&amp;nbsp;</code> for spaces.
</p>
"""