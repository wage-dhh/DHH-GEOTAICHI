from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from html import escape
from PIL import Image
import csv
import numpy as np

ROOT=Path(r'F:/dhh/GeoTaichi-dhh')
SCRIPT=ROOT/'example/mpm/FLAC3D/example3_3_free_field_shear_wave.py'
RESULT=ROOT/'example/mpm/FLAC3D/results/example3_3_free_field_shear_wave'
REPORT=RESULT/'report'
DOCX=REPORT/'FLAC3D_Example_3_3_complete_reproduction_report.docx'
summary={}
for line in (RESULT/'validation_summary.txt').read_text(encoding='utf-8').splitlines()[1:]:
    if ' = ' in line:
        k,v=line.split(' = ',1); summary[k]=v
with (RESULT/'histories.csv').open(encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
first=rows[0]; last=rows[-1]
imgs=[REPORT/'fig4_model.png',REPORT/'fig1_input_wave.png',REPORT/'fig2_x_velocity.png',REPORT/'fig3_response_checks.png']

def esc(s): return escape(str(s), quote=False)
def run(t,size=22,bold=False,color='000000'):
    b='<w:b/><w:bCs/>' if bold else ''
    return f'<w:r><w:rPr>{b}<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/><w:color w:val="{color}"/></w:rPr><w:t xml:space="preserve">{esc(t)}</w:t></w:r>'
def p(t='',size=22,bold=False,color='000000',after=120,jc=None):
    j=f'<w:jc w:val="{jc}"/>' if jc else ''
    return f'<w:p><w:pPr><w:spacing w:after="{after}" w:line="264" w:lineRule="auto"/>{j}</w:pPr>{run(t,size,bold,color)}</w:p>'
def h1(t): return p(t,size=32,bold=True,color='2E74B5',after=160)
def h2(t): return p(t,size=26,bold=True,color='1F4D78',after=120)
def kv(k,v): return p(f'{k}: {v}',size=21,after=60)
def bullet(t): return p('• '+t,size=21,after=80)
rels=[]
def image(path, caption, width_in=6.15):
    rid=f'rId{len(rels)+1}'; rels.append((rid,path))
    im=Image.open(path); cx=int(width_in*914400); cy=int(cx*im.height/im.width); im.close(); did=len(rels)
    xml=f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:after="80"/></w:pPr><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{did}" name="{esc(path.name)}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{did}" name="{esc(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    return xml+p(caption,size=18,color='555555',after=160,jc='center')

body=[]
body.append(p('FLAC3D Example 3.3 自由场边界剪切波算例完整复现报告',size=34,bold=True,color='0B2545',after=80,jc='center'))
body.append(p('GeoTaichi MPM | 文档参数复现 | 单点监测 | 结果对比',size=21,color='555555',after=220,jc='center'))
body.append(h1('0. 报告摘要'))
for k,v in [('复现对象','FLAC3D 3.1 Dynamic Analysis Example 3.3: Shear wave loading of a model with free-field boundaries'),('文档来源','C:/Users/Dell/Desktop/example3.3.pdf'),('代码文件',str(SCRIPT.relative_to(ROOT)).replace('\\','/')),('结果目录',str(RESULT.relative_to(ROOT)).replace('\\','/')),('运行时间',summary.get('simulation_time','0.015')+' s'),('物质点数量',summary.get('particle_number','120')),('历史记录数量',summary.get('history_rows','105'))]: body.append(kv(k,v))
body.append(h1('1. 原 FLAC3D 算例说明'))
body.append(p('Example 3.3 用于演示 FLAC3D 自由场边界。模型底部施加半余弦剪切应力波，计算 0.015 s，并在主网格、自由场侧边和角点位置记录顶部 x 速度历史。'))
body.append(h2('1.1 原始命令核心内容'))
for line in ['per = 0.01','wave = 0.5 * (1.0 - cos(2*pi*dytime/per))','gen zone brick size 6 3 2','gen zone brick size 2 3 2 p0 0 0 2','gen zone brick size 2 3 2 p0 4 0 2','gen zone wedge size 1 3 2 p0 2 0 2','gen zone wedge size 1 3 2 p0 4 3 2 ...','model elastic prop bulk 66667 shear 40000 ini dens 0.0025','set grav 0 0 -10','apply dstress 1.0 hist wave range z -0.1 0.1','apply ff','solve age 0.015']:
    body.append(p(line,size=18,after=40))
body.append(h1('2. GeoTaichi MPM 复现方法'))
body.append(h2('2.1 维度转换'))
body.append(p('FLAC3D 原模型为三维，但几何和加载在 y 方向均匀。本复现采用二维 x-z 剖面，GeoTaichi 的二维坐标 x-y 对应 FLAC3D 的 x-z。'))
body.append(h2('2.2 几何复现'))
body.append(p('主网格由底部矩形、左右上部矩形和两个三角楔体组成；两侧增加显式自由场柱，用于记录自由场顶部响应。脚本在运行前根据几何方程生成物质点文件 particles_example3_3.txt。'))
body.append(image(imgs[0],'图 1  GeoTaichi MPM 几何与加载示意图。'))
body.append(h2('2.3 材料参数换算'))
for k,v in [('FLAC3D bulk modulus','66667'),('FLAC3D shear modulus','40000'),('FLAC3D density','0.0025'),('MPM Young modulus',summary.get('young_modulus','')),('MPM Poisson ratio',summary.get('poisson_ratio','')),('Shear wave velocity',summary.get('shear_wave_velocity','')),('P wave velocity',summary.get('p_wave_velocity',''))]: body.append(kv(k,v))
body.append(h2('2.4 动态加载'))
body.append(p('底部 x 向剪切加载按 FLAC3D 文档的 apply dstress 1.0 hist wave 实现。每个时间步回调更新底部 TractionConstraint，使 bottom_dstress 与 wave(t) 一致。'))
body.append(image(imgs[1],'图 2  FLAC3D wave(t) 与 MPM bottom_dstress 输入对比。'))
body.append(h1('3. 单点监测设置'))
body.append(p('监测方式已经由区域平均改为单个物质点监测。脚本给定目标坐标，运行时选取离目标坐标最近的物质点，输出该物质点的速度、加速度和剪应力。'))
for k in ['main_top','left_ff_top','right_ff_top','base']:
    body.append(kv(k+' 实际点坐标', f"({first.get(k+'_point_x','')}, {first.get(k+'_point_y','')})"))
body.append(p('输出字段包含 point_x 和 point_y，用于确认实际选中的物质点位置。例如 main_top_point_x/main_top_point_y、base_point_x/base_point_y。'))
body.append(h1('4. 结果对比'))
body.append(h2('4.1 x 速度历史'))
body.append(p('FLAC3D Figure 3.9 的主要比较对象是顶部 x 速度历史。图 3 给出 MPM 中主网格顶部、左右自由场顶部和底部单点监测的 x 速度历史。'))
body.append(image(imgs[2],'图 3  单点监测 x 速度历史。'))
body.append(h2('4.2 加速度与剪应力检查'))
body.append(p('图 4 给出 x 加速度和 tau_xy 剪应力历史，用于检查动态输入在模型中的传播与响应量级。'))
body.append(image(imgs[3],'图 4  单点监测加速度与剪应力响应。'))
body.append(h1('5. 运行输出文件'))
for k,v in [('histories.csv','单点监测历史；包含目标点实际坐标、vx、vy、ax、tau_xy'),('validation_summary.txt','材料、波速、时间、粒子数和输出路径摘要'),('grids/*.npz','MPM 背景网格保存数据'),('vtks/*.vtr','后处理可视化网格'),('report/*.docx, *.pdf','本复现报告及导出 PDF')]: body.append(kv(k,v))
body.append(h1('6. 结论'))
body.append(bullet('已按 FLAC3D Example 3.3 文档参数完整复现材料、几何、重力、波形、底部剪切应力加载和 0.015 s 动力计算。'))
body.append(bullet('监测方式已改为单个物质点监测，避免区域平均掩盖局部响应。'))
body.append(bullet('输入波形与文档函数一致；输出历史可用于与 FLAC3D Figure 3.9 数字化曲线进一步逐点对比。'))
body.append(h1('7. 局限性'))
body.append(bullet('GeoTaichi 当前没有与 FLAC3D apply ff 和 nquiet/squiet/dquiet 完全等价的命令级接口，因此这里采用显式自由场柱和底部牵引边界近似。'))
body.append(bullet('若需要严格误差指标，需要从 FLAC3D 文档图中数字化原始曲线，或获取 FLAC3D 原始 history 文件。'))
body.append(h1('8. 复现命令'))
body.append(p("& 'D:\\self_programs\\anaconda\\ana\\envs\\env_rec\\python.exe' .\\example\\mpm\\FLAC3D\\example3_3_free_field_shear_wave.py",size=18,after=80))
body.append(p('可设置 EXAMPLE3_3_SMOKE=1 进行短时装配检查；默认不设置时运行完整 0.015 s。',size=20,after=120))
sect='<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
doc=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>{"".join(body)}{sect}</w:body></w:document>'
ct='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
rootrels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
drels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>' for rid,path in rels)+'</Relationships>'
with ZipFile(DOCX,'w',ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml',ct); z.writestr('_rels/.rels',rootrels); z.writestr('word/document.xml',doc); z.writestr('word/_rels/document.xml.rels',drels)
    for _,path in rels: z.write(path,f'word/media/{path.name}')
print(DOCX)
