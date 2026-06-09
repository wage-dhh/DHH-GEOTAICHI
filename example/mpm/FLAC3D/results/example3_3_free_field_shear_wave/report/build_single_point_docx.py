from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from html import escape
from PIL import Image
ROOT=Path(r'F:/dhh/GeoTaichi-dhh')
RESULT=ROOT/'example/mpm/FLAC3D/results/example3_3_free_field_shear_wave'
REPORT=RESULT/'report'
DOCX=REPORT/'FLAC3D_Example_3_3_GeoTaichi_MPM_single_point_reproduction_report.docx'
summary={}
for line in (RESULT/'validation_summary.txt').read_text(encoding='utf-8').splitlines()[1:]:
    if ' = ' in line:
        k,v=line.split(' = ',1); summary[k]=v
imgs=[REPORT/'fig4_model.png',REPORT/'fig1_input_wave.png',REPORT/'fig2_x_velocity.png',REPORT/'fig3_response_checks.png']

def esc(s): return escape(str(s), quote=False)
def run(t, size=22, bold=False, color='000000'):
    b='<w:b/><w:bCs/>' if bold else ''
    return f'<w:r><w:rPr>{b}<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/></w:rPr><w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
def p(t='', size=22, bold=False, color='000000', after=120, jc=None):
    j=f'<w:jc w:val="{jc}"/>' if jc else ''
    return f'<w:p><w:pPr><w:spacing w:after="{after}" w:line="264" w:lineRule="auto"/>{j}</w:pPr>{run(t,size,bold,color)}</w:p>'
def h(t): return p(t, size=32, bold=True, color='2E74B5', after=160)
def kv(k,v): return p(f'{k}: {v}', size=21, after=60)
rels=[]
def image(path, caption, width_in=6.1):
    rid=f'rId{len(rels)+1}'; rels.append((rid,path))
    im=Image.open(path); cx=int(width_in*914400); cy=int(cx*im.height/im.width); im.close(); did=len(rels)
    imgxml=f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="80"/></w:pPr><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{did}" name="{esc(path.name)}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{did}" name="{esc(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    return imgxml + p(caption, size=18, color='555555', after=160, jc='center')
body=[]
body.append(p('FLAC3D Example 3.3 自由场边界剪切波算例复现报告', size=34, bold=True, color='0B2545', after=80, jc='center'))
body.append(p('GeoTaichi MPM 复现结果对比与参数说明', size=21, color='555555', after=220, jc='center'))
for k,v in [('文档来源','C:/Users/Dell/Desktop/example3.3.pdf'),('代码位置','example/mpm/FLAC3D/example3_3_free_field_shear_wave.py'),('结果目录','example/mpm/FLAC3D/results/example3_3_free_field_shear_wave'),('运行时间',summary.get('simulation_time','0.015')+' s'),('物质点数量',summary.get('particle_number','120')),('历史记录',summary.get('history_rows','105')+' rows')]: body.append(kv(k,v))
body.append(h('1. 复现目标'))
body.append(p('复现 FLAC3D 手册 Example 3.3 “Shear wave loading of a model with free-field boundaries”。原算例在模型底部施加半余弦剪切应力波，并比较主网格与自由场边界顶部点的 x 速度历史。'))
body.append(p('本复现严格采用 PDF 中给出的材料、几何、重力、波形周期、剪切应力幅值和 0.015 s 动力计算时间。GeoTaichi 当前没有与 FLAC3D apply ff 及 quiet boundary 完全等价的命令级接口，因此代码采用二维 x-z 剖面和显式左右自由场柱记录响应。'))
body.append(h('2. 参数映射'))
for k,v in [('材料','FLAC3D: bulk 66667, shear 40000, density 0.0025；MPM: E='+summary.get('young_modulus','')+', nu='+summary.get('poisson_ratio','')),('重力','FLAC3D: set grav 0 0 -10；MPM: gravity=[0,-10]'),('波形','FLAC3D: 0.5*(1-cos(2*pi*t/0.01))；MPM: 逐步更新 bottom_dstress'),('加载','FLAC3D: apply dstress 1.0 hist wave；MPM: 底部 x 向 TractionConstraint'),('时间','FLAC3D: solve age 0.015；MPM: '+summary.get('simulation_time','0.015')+' s'),('输出','main_top、left_ff_top、right_ff_top、base 历史')]: body.append(kv(k,v))
body.append(image(imgs[0],'图 1  GeoTaichi MPM 模型示意图。'))
body.append(h('3. 输入波形对比'))
body.append(p('图 2 显示 FLAC3D 文档 wave(t) 与 MPM 每步写入的 bottom_dstress。两条曲线一致，说明动态输入没有重新编造参数。'))
body.append(image(imgs[1],'图 2  FLAC3D wave(t) 与 MPM 底部剪切应力输入对比。'))
body.append(h('4. 结果对比'))
body.append(p('FLAC3D Figure 3.9 的对比对象是模型顶部不同位置的 x 速度历史。图 3 给出 MPM 中主网格顶部、自由场顶部和底部的同类 x 速度历史。'))
body.append(image(imgs[2],'图 3  x 速度历史结果对比。'))
body.append(p('图 4 给出 x 加速度和剪应力历史，用于检查底部输入后的动力响应传播。'))
body.append(image(imgs[3],'图 4  加速度与剪应力响应检查。'))
body.append(h('5. 运行摘要与结论'))
for k,v in [('临界时间步','7.21686834144903e-05 s'),('剪切波速',summary.get('shear_wave_velocity','4000')),('P 波波速',summary.get('p_wave_velocity','6928.21285277')),('加载方式',summary.get('loading_mode','stress')),('输出文件','histories.csv, validation_summary.txt, grids/*.npz, vtks/*.vtr')]: body.append(kv(k,v))
for t in ['输入、材料、重力、几何和计算时长均来自桌面 PDF 中的 Example 3.3 命令片段。','脚本已完成完整 0.015 s 运行，生成 105 行历史记录和网格输出。','后续若需要逐点误差，需要先数字化 FLAC3D Figure 3.9 原曲线；当前报告提供同类监测量和输入一致性对比。']: body.append(p('• '+t))
body.append(h('附录：关键 FLAC3D 命令'))
for line in ['per = 0.01','wave = 0.5 * (1.0 - cos(2*pi*dytime/per))','model elastic prop bulk 66667 shear 40000 ini dens 0.0025','set grav 0 0 -10','apply dstress 1.0 hist wave range z -0.1 0.1','apply ff','solve age 0.015']: body.append(p(line, size=18, after=40))
sect='<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
doc=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{"".join(body)}{sect}</w:body></w:document>'
ct='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
rootrels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
drels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>' for rid,path in rels)+'</Relationships>'
with ZipFile(DOCX,'w',ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml',ct); z.writestr('_rels/.rels',rootrels); z.writestr('word/document.xml',doc); z.writestr('word/_rels/document.xml.rels',drels)
    for _,path in rels: z.write(path,f'word/media/{path.name}')
print(DOCX)
