import branca
import matplotlib as mpl
import matplotlib.pyplot as plt

def POI_legend(labels, norm):
    
    target1 = '{% macro html(this, kwargs) %}'
    target2 = '{% endmacro %}'

    legend_list = ''
    for value in labels:
        color = plt.cm.viridis(norm(value))
        hex_color = mpl.colors.to_hex(color)
        legend_tmp = f'<p><a style="color:{hex_color};font-size:200%;margin-left:20px;line-height:0px">&diams;</a>&emsp;{value}</p>'
        legend_list = legend_list + legend_tmp

        legend_html = f'''
            {target1}
            <div style="
                position: fixed; 
                bottom: 30px;
                right: 20px;
                width: 150px;
                height: 260px;
                z-index:9999;
                font-size:20px;
                ">
                {legend_list}
            </div>
            {target2}
            '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    return legend

def Sem_legend(labels, norm):
    
    target1 = '{% macro html(this, kwargs) %}'
    target2 = '{% endmacro %}'

    legend_list = ''
    for value in labels:
        color = plt.cm.Spectral(norm(value))
        hex_color = mpl.colors.to_hex(color)
        legend_tmp = f'<p><a style="color:{hex_color};font-size:200%;margin-left:20px;line-height:0px">&diams;</a>&emsp;{int(value)}</p>'
        legend_list = legend_list + legend_tmp

        legend_html = f'''
            {target1}
            <div style="
                position: fixed; 
                bottom: 30px;
                right: 1px;
                width: 150px;
                height: 260px;
                z-index:9999;
                font-size:20px;
                ">
                {legend_list}
            </div>
            <div style="
                position: fixed;
                bottom: 10px;
                right: 40px;
                width: 100px;
                height: 300px; 
                z-index:9998;
                font-size:14px;
                background-color: #fefefe;

                opacity: 0.7;
                ">
            </div>
            {target2}
            '''
    legend = branca.element.MacroElement()
    legend._template = branca.element.Template(legend_html)
    return legend


class StrLabel:
    def str_label_1(
        BOOL_AOI = False,
        BOOL_POI = False,
        BOOL_SEM = False,
        ):
        font_color = 'black'
        if BOOL_AOI:
            MODEL_AOI = f'<b style="color: {font_color};">{BOOL_AOI}</b>'
        else:
            MODEL_AOI = BOOL_AOI
        if BOOL_POI:
            MODEL_POI = f'<b style="color: {font_color};">{BOOL_POI}</b>'
        else:
            MODEL_POI = BOOL_POI
        if BOOL_SEM:
            MODEL_SEM = f'<b style="color: {font_color};">{BOOL_SEM}</b>'
        else:
            MODEL_SEM = BOOL_SEM
            
        STR_LABEL_1 = f'''
            <p><b style="color: black; font-size: 32px;">Data Status:</b> 
            <br>AOI Shp:{MODEL_AOI}
            <br>POI Shp:{MODEL_POI}
            <br>LC  Shp:{MODEL_SEM}
            </p>
        '''
        return STR_LABEL_1

    def str_label_2(
        BOOL_SEMGRAPH_MST = False,
        BOOL_SEMGRAPH_DT  = False,
        BOOL_SEMGRAPH_IDT = False,
        BOOL_PHYGRAPH_MST = False,
        BOOL_PHYGRAPH_DT  = False,
        BOOL_PHYGRAPH_IDT = False,
        ):
        font_color = 'black'
        if BOOL_SEMGRAPH_MST:
            MODEL_SEMGRAPH_MST = f'<b style="color: {font_color};">{BOOL_SEMGRAPH_MST}</b>'
        else:
            MODEL_SEMGRAPH_MST = BOOL_SEMGRAPH_MST
        if BOOL_SEMGRAPH_DT:
            MODEL_SEMGRAPH_DT = f'<b style="color: {font_color};">{BOOL_SEMGRAPH_DT}</b>'
        else:
            MODEL_SEMGRAPH_DT = BOOL_SEMGRAPH_DT
        if BOOL_SEMGRAPH_IDT:
            MODEL_SEMGRAPH_IDT = f'<b style="color: {font_color};">{BOOL_SEMGRAPH_IDT}</b>'
        else:
            MODEL_SEMGRAPH_IDT = BOOL_SEMGRAPH_IDT
        if BOOL_PHYGRAPH_MST:
            MODEL_PHYGRAPH_MST = f'<b style="color: {font_color};">{BOOL_PHYGRAPH_MST}</b>'
        else:
            MODEL_PHYGRAPH_MST = BOOL_PHYGRAPH_MST
        if BOOL_PHYGRAPH_DT:
            MODEL_PHYGRAPH_DT = f'<b style="color: {font_color};">{BOOL_PHYGRAPH_DT}</b>'
        else:
            MODEL_PHYGRAPH_DT = BOOL_PHYGRAPH_DT
        if BOOL_PHYGRAPH_IDT:
            MODEL_PHYGRAPH_IDT = f'<b style="color: {font_color};">{BOOL_PHYGRAPH_IDT}</b>'
        else:
            MODEL_PHYGRAPH_IDT = BOOL_PHYGRAPH_IDT
        STR_LABEL_2 = f'''
            <p><b style="color: black; font-size: 32px;">Graph Status:</b> 
            <br>SemGraph_MST:{MODEL_SEMGRAPH_MST}
            <br>SemGraph_DT :{MODEL_SEMGRAPH_DT}
            <br>SemGraph_IDT:{MODEL_SEMGRAPH_IDT}
            <br>PhyGraph_MST:{MODEL_PHYGRAPH_MST}
            <br>PhyGraph_DT :{MODEL_PHYGRAPH_DT}
            <br>PhyGraph_IDT:{MODEL_PHYGRAPH_IDT}
            </p>
        '''
        return STR_LABEL_2

    def str_label_3(
        BOOL_G2MF_MODEL= False,
        BOOL_SEM_MODEL = False,
        BOOL_PHY_MODEL = False,
        BOOL_OWN_MODEL = False,
        ):
        font_color = 'black'
        if BOOL_G2MF_MODEL:
            MODEL_G2MF_MODEL = f'<b style="color: {font_color};">{BOOL_G2MF_MODEL}</b>'
        else:
            MODEL_G2MF_MODEL = BOOL_G2MF_MODEL
        if BOOL_SEM_MODEL:
            MODEL_SEM_MODEL = f'<b style="color: {font_color};">{BOOL_SEM_MODEL}</b>'
        else:
            MODEL_SEM_MODEL = BOOL_SEM_MODEL
        if BOOL_PHY_MODEL:
            MODEL_PHY_MODEL = f'<b style="color: {font_color};">{BOOL_PHY_MODEL}</b>'
        else:
            MODEL_PHY_MODEL = BOOL_PHY_MODEL
        if BOOL_OWN_MODEL:
            MODEL_OWN_MODEL = f'<b style="color: {font_color};">{BOOL_OWN_MODEL}</b>'
        else:
            MODEL_OWN_MODEL = BOOL_OWN_MODEL
        STR_LABEL_3 = f'''
            <p><b style="color: black; font-size: 32px;">Model Status:</b> 
            <br>G2MF_Model:{MODEL_G2MF_MODEL}
            <br>Sem_Model:{MODEL_SEM_MODEL}
            <br>Phy_Model:{MODEL_PHY_MODEL}
            <br>OWN_Model:{MODEL_OWN_MODEL}
            </p>
        '''
        return STR_LABEL_3
    
    def str_label_4(
        TYPE_GT       = '?',
        TYPE_G2MF_MST = '?',
        TYPE_G2MF_DT  = '?',
        TYPE_G2MF_IDT = '?',
        TYPE_SEM_MST  = '?',
        TYPE_SEM_DT   = '?',
        TYPE_SEM_IDT  = '?',
        TYPE_PHY_MST  = '?',
        TYPE_PHY_DT   = '?',
        TYPE_PHY_IDT  = '?',
        ):
        STR_LABEL_4 = f'''
            <p><b style="color: black; font-size: 32px;">Inference:</b> 
            <br>GroundTruth:{TYPE_GT}
            <br>G2MF_MST:{TYPE_G2MF_MST}
            <br>G2MF_DT: {TYPE_G2MF_DT}
            <br>G2MF_IDT:{TYPE_G2MF_IDT}
            <br>Sem_MST:{TYPE_SEM_MST}
            <br>Sem_DT: {TYPE_SEM_DT}
            <br>Sem_IDT:{TYPE_SEM_IDT}
            <br>Phy_MST:{TYPE_PHY_MST}
            <br>Phy_DT: {TYPE_PHY_DT}
            <br>Phy_IDT:{TYPE_PHY_IDT}
            </p>
        '''
        return STR_LABEL_4

