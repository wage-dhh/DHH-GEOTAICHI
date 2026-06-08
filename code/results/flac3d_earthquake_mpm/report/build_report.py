from pathlib import Path
import csv
import zipfile
from datetime import datetime, timezone
from html import escape

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(r'F:/dhh/GeoTaichi-dhh')
RESULT_DIR = ROOT / 'code/results/flac3d_earthquake_mpm'
REPORT_DIR = RESULT_DIR / 'report'
REPORT_DIR.mkdir(parents=True, exist_ok=True)
DOCX_PATH = REPORT_DIR / 'FLAC3D_earthquake_MPM_reproduction_report.docx'

DOC_ACCEL_RANGES = {'base': (-8.656, 8.199), 'middle': (-17.99, 13.77), 'surface': (-18.11, 17.09)}
DOC_STRESS_STRAIN_RANGES = {
    'base': {'strain': (-1.315e-3, 1.988e-3), 'stress': (-2.413e5, 3.079e5)},
    'middle': {'strain': (-6.766e-4, 8.415e-4), 'stress': (-1.589e5, 1.824e5)},
}

def read_csv(path):
    with path.open('r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in rows[0].keys()}

hist = read_csv(RESULT_DIR / 'histories.csv')
motion = read_csv(RESULT_DIR / 'base_motion.csv')
summary = (RESULT_DIR / 'validation_summary.txt').read_text(encoding='utf-8')
summary_map = {}
for line in summary.splitlines()[1:]:
    if ' = ' in line:
        k, v = line.split(' = ', 1)
        summary_map[k.strip()] = v.strip()

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def save_figures():
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=220)
    ax.plot(motion['time'], motion['base_acceleration'], color='#B23A2E', lw=1.2, label='Function base acceleration')
    ax.axhspan(DOC_ACCEL_RANGES['base'][0], DOC_ACCEL_RANGES['base'][1], color='#B23A2E', alpha=0.10, label='lizhi base range')
    ax.set_xlabel('Time t / s'); ax.set_ylabel('Acceleration ax / m/s^2')
    ax.grid(True, color='#D7DDE5', lw=0.45); ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Function seismic input fitted from the final lizhi waveform figures')
    fig.tight_layout(); p1 = REPORT_DIR / 'fig1_input_wave.png'; fig.savefig(p1, bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2), dpi=220, sharex=True)
    series = [('base', 'Base', '#B23A2E'), ('middle', 'Middle', '#1F6F8B'), ('surface', 'Surface', '#2D7D46')]
    for ax, (key, label, color) in zip(axes, series):
        lo, hi = DOC_ACCEL_RANGES[key]
        ax.axhspan(lo, hi, color=color, alpha=0.10, label=f'lizhi range [{lo:g}, {hi:g}]')
        ax.plot(hist['time'], hist[f'{key}_ax'], color=color, lw=0.9, label='MPM response')
        if key == 'base':
            ax.plot(hist['time'], hist['input_base_ax'], color='#111111', lw=0.65, alpha=0.75, label='input acceleration')
        ax.set_ylabel(f'{label}\nax'); ax.grid(True, color='#D7DDE5', lw=0.45); ax.legend(loc='upper right', fontsize=7)
    axes[-1].set_xlabel('Time t / s')
    fig.suptitle('Acceleration comparison: MPM response vs ranges from lizhi result figures', y=0.995, fontsize=11)
    fig.tight_layout(); p2 = REPORT_DIR / 'fig2_acceleration_comparison.png'; fig.savefig(p2, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=220)
    for key, label, color in [('base', 'Base', '#B23A2E'), ('middle', 'Middle', '#1F6F8B')]:
        ranges = DOC_STRESS_STRAIN_RANGES[key]
        x0, x1 = ranges['strain']; y0, y1 = ranges['stress']
        ax.fill_between([x0, x1], [y0, y0], [y1, y1], color=color, alpha=0.12, label=f'lizhi {label} range')
        ax.plot(hist[f'{key}_strain'], hist[f'{key}_shear_stress'], color=color, lw=0.9, label=f'MPM {label}')
    ax.axhline(0, color='#666666', lw=0.6); ax.axvline(0, color='#666666', lw=0.6)
    ax.set_xlabel('Shear strain gamma'); ax.set_ylabel('Shear stress tau / Pa')
    ax.grid(True, color='#D7DDE5', lw=0.45); ax.legend(loc='best', fontsize=8)
    ax.set_title('Shear stress-strain comparison: MPM path vs lizhi result ranges')
    fig.tight_layout(); p3 = REPORT_DIR / 'fig3_stress_strain_comparison.png'; fig.savefig(p3, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=220)
    for key, label, color in series:
        ax.plot(hist['time'], hist[f'{key}_xdisp'], color=color, lw=1.0, label=label)
    ax.set_xlabel('Time t / s'); ax.set_ylabel('Horizontal displacement ux / m')
    ax.grid(True, color='#D7DDE5', lw=0.45); ax.legend(loc='upper right', fontsize=8)
    ax.set_title('MPM horizontal displacement histories')
    fig.tight_layout(); p4 = REPORT_DIR / 'fig4_displacement.png'; fig.savefig(p4, bbox_inches='tight'); plt.close(fig)
    return p1, p2, p3, p4

input_png, acc_png, stress_png, disp_png = save_figures()

NS = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','wp':'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing','a':'http://schemas.openxmlformats.org/drawingml/2006/main','pic':'http://schemas.openxmlformats.org/drawingml/2006/picture'}
def x(s): return escape(str(s), quote=False)
def run(text, bold=False, italic=False, size=22, color='000000'):
    flags = ('<w:b/><w:bCs/>' if bold else '') + ('<w:i/><w:iCs/>' if italic else '')
    return f'<w:r><w:rPr>{flags}<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/></w:rPr><w:t xml:space="preserve">{x(text)}</w:t></w:r>'
def para(text='', style=None, runs=None, before=None, after=120, line=264, jc=None):
    ppr = []
    if style: ppr.append(f'<w:pStyle w:val="{style}"/>')
    if before is not None or after is not None or line is not None:
        bits = []
        if before is not None: bits.append(f'w:before="{before}"')
        if after is not None: bits.append(f'w:after="{after}"')
        if line is not None: bits.append(f'w:line="{line}" w:lineRule="auto"')
        ppr.append(f'<w:spacing {" ".join(bits)}/>')
    if jc: ppr.append(f'<w:jc w:val="{jc}"/>')
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{"".join(runs) if runs else run(text)}</w:p>'
def heading(text, level): return para(text, style=f'Heading{level}', after=120)
def bullet(text): return f'<w:p><w:pPr><w:pStyle w:val="ListBullet"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr><w:spacing w:after="160" w:line="280" w:lineRule="auto"/></w:pPr>{run(text)}</w:p>'
def caption(text): return para(runs=[run(text, italic=True, size=18, color='555555')], after=160, jc='center')
def table(rows, widths):
    grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths); trs=[]
    for ri,row in enumerate(rows):
        cells=[]
        for ci,cell in enumerate(row):
            fill = '<w:shd w:fill="F2F4F7"/>' if ri==0 else ''
            cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{fill}<w:tcMar><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tcMar></w:tcPr>{para(str(cell), after=40, line=260)}</w:tc>')
        trs.append('<w:tr>'+''.join(cells)+'</w:tr>')
    return '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D0D7DE"/><w:left w:val="single" w:sz="4" w:color="D0D7DE"/><w:bottom w:val="single" w:sz="4" w:color="D0D7DE"/><w:right w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideH w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideV w:val="single" w:sz="4" w:color="D0D7DE"/></w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>'+grid+'</w:tblGrid>'+''.join(trs)+'</w:tbl>'
image_rels=[]
def image_para(path, width_in=6.2):
    rid=f'rIdImg{len(image_rels)+1}'; image_rels.append((rid,path))
    with Image.open(path) as im: wpx,hpx=im.size
    cx=int(width_in*914400); cy=int(cx*hpx/wpx); did=len(image_rels)
    return f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="80"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{did}" name="{x(path.name)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{did}" name="{x(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'

body=[]
body.append(para(runs=[run('FLAC3D 地震模块算例的 GeoTaichi MPM 复现报告', size=36, bold=True, color='0B2545')], after=80))
body.append(para(runs=[run('对象：桌面 lizhi 文档中的一维弹性土层地震输入算例；实现脚本：PIPLE.py；计算结果目录：code/results/flac3d_earthquake_mpm。', size=20, color='555555')], after=220))
body.append(table([['项目','值'],['报告生成时间',datetime.now().strftime('%Y-%m-%d %H:%M')],['计算总时长',summary_map.get('final_time','25 s')],['时间步长',summary_map.get('dt','2e-4')],['历史记录行数',summary_map.get('history_rows','1252')],['波形来源',summary_map.get('input_source','函数波形')]], [2200,7160]))
body.append(heading('1. 复现目标',1))
body.append(para('本报告说明如何用现有 GeoTaichi MPM 代码复现 FLAC3D 地震模块文档中的一维弹性土层算例。原 FLAC3D 算例关注水平地震输入在均匀弹性层中的传播，并输出底部、中部和自由表面的加速度时程，以及底部和中部的剪应力-剪应变响应。'))
body.append(para('本次复现采用二维 MPM 剪切柱模型。由于当前要求不再依赖外部波形文件，脚本将 lizhi 文档最后两张波形图片对应的输入形态用解析函数代替，并按文档给出的基底加速度峰值范围进行标定。'))
body.append(heading('2. 模型与参数',1))
body.append(table([['参数','数值','说明'],['土层高度','20 m','与 FLAC3D 文档一维弹性层一致'],['计算宽度','1 m','二维剪切柱，横向只保留一列单元'],['单元尺寸','1 m x 1 m','Q4N2D 单元'],['密度','1000 kg/m3','文档参数'],['剪切模量 G','5.0e8 Pa','文档参数'],['体积模量 K','1.0e9 Pa','文档参数'],['等效 Young 模量',summary_map.get('young_modulus',''),'由 K 与 G 换算'],['Poisson 比',summary_map.get('poisson_ratio',''),'由 K 与 G 换算'],['剪切波速',summary_map.get('shear_wave_velocity',''),'sqrt(G/rho)']], [2300,2200,4860]))
body.append(heading('3. 地震波函数替代',1))
body.append(para('PIPLE.py 中使用 8 个局部 Gabor 脉冲叠加构造基底加速度函数。该函数不是读取 gilroy1.acc，而是直接在代码中由中心时刻、持续宽度、频率、幅值和相位控制，随后缩放到 lizhi 文档结果图中基底加速度范围 [-8.656, 8.199] m/s2。'))
body.append(para('函数型输入的优点是复现过程不依赖外部数据文件，缺点是它只等效匹配文档图片的主要波形包络与峰值范围，不能代表原始 FLAC3D 记录逐点一致。'))
body.append(image_para(input_png)); body.append(caption('图 1  函数型基底加速度输入及 lizhi 文档基底加速度范围'))
body.append(heading('4. 边界条件与求解流程',1))
body.append(bullet('底部水平速度：由函数加速度经梯形积分得到，并在每个 MPM 时间步动态更新底部 x 向速度约束。'))
body.append(bullet('全高竖向速度：设置 y 向速度为 0，形成一维剪切传播条件。'))
body.append(bullet('重力：关闭，以便对应文档中的均匀弹性层动力响应验证。'))
body.append(bullet('输出：记录底部、中部、表面的速度、加速度、位移，以及底部和中部的剪应力-剪应变路径。'))
body.append(heading('5. 结果图片对比',1))
body.append(para('图 2 将 MPM 计算得到的三处加速度时程与 lizhi 文档结果图片中给出的加速度范围叠加显示。阴影区表示文档图中的结果范围，曲线表示本次 MPM 计算响应。'))
body.append(image_para(acc_png,6.25)); body.append(caption('图 2  底部、中部和表面加速度时程对比'))
body.append(para('图 3 给出底部和中部剪应力-剪应变轨迹。矩形阴影表示 lizhi 文档结果图中的剪应变和剪应力范围，曲线为 MPM 输出历史。该图用于检查响应量级、滞回路径范围和模型动力输入是否处在合理区间。'))
body.append(image_para(stress_png,6.25)); body.append(caption('图 3  剪应力-剪应变结果图片对比'))
body.append(para('图 4 展示三处监测点的水平位移响应。该图不是 FLAC3D 文档主要对比图，但有助于判断输入积分后的基底速度是否引入异常长期漂移。'))
body.append(image_para(disp_png,6.25)); body.append(caption('图 4  MPM 水平位移响应'))
body.append(heading('6. 结果文件与复现命令',1))
body.append(table([['文件','用途'],['PIPLE.py','GeoTaichi MPM 复现脚本，包含函数型地震输入'],['base_motion.csv','函数基底加速度和积分速度'],['histories.csv','底部、中部、表面响应历史'],['acceleration_comparison.svg','加速度时程对比图'],['stress_strain_comparison.svg','剪应力-剪应变对比图'],['validation_summary.txt','关键参数和运行摘要']], [3400,5960]))
body.append(para('复现命令：'))
body.append(para(r"& 'D:\self_programs\anaconda\ana\envs\env_rec\python.exe' PIPLE.py", runs=[run(r"& 'D:\self_programs\anaconda\ana\envs\env_rec\python.exe' PIPLE.py", size=20, color='0B2545')], before=40, after=120))
body.append(heading('7. 结论与局限',1))
body.append(bullet('现有 MPM 框架能够复现 FLAC3D 地震模块算例的核心流程：材料换算、基底动力输入、监测点响应输出和结果图对比。'))
body.append(bullet('函数型波形已按 lizhi 文档最后结果图片的基底加速度范围标定，适合用于无外部波形文件条件下的可重复验证。'))
body.append(bullet('若后续需要与 FLAC3D 原始结果逐点误差对比，仍需要从文档或原 FLAC3D 工程中恢复原始地震记录和图中曲线数据。'))

sect='<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/><w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
document_xml=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{NS["w"]}" xmlns:r="{NS["r"]}" xmlns:wp="{NS["wp"]}" xmlns:a="{NS["a"]}" xmlns:pic="{NS["pic"]}"><w:body>{"".join(body)}{sect}</w:body></w:document>'
styles_xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:bCs/><w:color w:val="2E74B5"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:bCs/><w:color w:val="2E74B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720" w:hanging="360"/><w:spacing w:after="160" w:line="280" w:lineRule="auto"/></w:pPr></w:style></w:styles>'
numbering_xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:tabs><w:tab w:val="num" w:pos="720"/></w:tabs><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="1"/></w:num></w:numbering>'
content_types='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/><Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'
root_rels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'
rels=['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>','<Relationship Id="rIdNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>','<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>']
for rid,path in image_rels: rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>')
doc_rels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>'
settings='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/><w:defaultTabStop w:val="720"/></w:settings>'
now=datetime.now(timezone.utc).isoformat()
core=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>FLAC3D 地震模块算例的 GeoTaichi MPM 复现报告</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>'
app='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex OOXML Builder</Application></Properties>'
with zipfile.ZipFile(DOCX_PATH,'w',zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml',content_types); z.writestr('_rels/.rels',root_rels); z.writestr('word/document.xml',document_xml); z.writestr('word/styles.xml',styles_xml); z.writestr('word/numbering.xml',numbering_xml); z.writestr('word/settings.xml',settings); z.writestr('word/_rels/document.xml.rels',doc_rels); z.writestr('docProps/core.xml',core); z.writestr('docProps/app.xml',app)
    for _,path in image_rels: z.write(path, f'word/media/{path.name}')
print(DOCX_PATH)
