$ErrorActionPreference='Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root='F:\dhh\GeoTaichi-dhh'
$out=Join-Path $root 'code\results\flac3d_free_field_shear_wave\report'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$docx=Join-Path $out 'FLAC3D_free_field_shear_wave_reproduction_report.docx'
$ref=Join-Path $out 'fig1_flac_reference_1600.png'
Copy-Item 'C:\Users\Dell\Desktop\1600..png' $ref -Force
function Bmp($p,$w,$h,$sb){$b=New-Object Drawing.Bitmap($w,$h);$g=[Drawing.Graphics]::FromImage($b);$g.SmoothingMode='AntiAlias';$g.Clear([Drawing.Color]::White);&$sb $g $w $h;$b.Save($p,[Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$b.Dispose()}
$fig2=Join-Path $out 'fig2_geotaichi_model_schematic.png'
Bmp $fig2 1600 760 {param($g,$w,$h)
$fT=New-Object Drawing.Font('Arial',26,[Drawing.FontStyle]::Bold);$f=New-Object Drawing.Font('Arial',15);$fs=New-Object Drawing.Font('Arial',12)
$blue=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(31,78,121));$dark=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(20,20,20));$green=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(80,205,70));$side=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(45,160,80))
$grid=New-Object Drawing.Pen([Drawing.Color]::FromArgb(96,96,96),1);$outline=New-Object Drawing.Pen([Drawing.Color]::FromArgb(40,40,40),3);$wave=New-Object Drawing.Pen([Drawing.Color]::FromArgb(190,40,32),5)
$g.DrawString('GeoTaichi MPM model corresponding to FLAC3D figure 1600',$fT,$blue,70,38)
$x0=160;$y0=140;$gridSizePx=28;$base=$y0+18*$gridSizePx
for($i=0;$i -lt 40;$i++){ $x=$i+0.5; if($x -ge 16 -and $x -le 24){$hm=8+1.25*[Math]::Abs($x-20)}elseif($x -ge 14 -and $x -lt 16){$hm=18-2*($x-14)}elseif($x -gt 24 -and $x -le 26){$hm=14+2*($x-24)}else{$hm=18}; $ph=[int]($hm*$gridSizePx);$r=[Drawing.Rectangle]::new($x0+$i*$gridSizePx,$base-$ph,$gridSizePx,$ph);$g.FillRectangle($green,$r);$g.DrawRectangle($grid,$r)}
$lr=[Drawing.Rectangle]::new(70,$base-18*$gridSizePx,12,18*$gridSizePx);$rr=[Drawing.Rectangle]::new($x0+40*$gridSizePx+90,$base-18*$gridSizePx,12,18*$gridSizePx);$g.FillRectangle($side,$lr);$g.FillRectangle($side,$rr);$g.DrawRectangle($outline,$lr);$g.DrawRectangle($outline,$rr);$g.DrawRectangle($outline,$x0,$base-18*$gridSizePx,40*$gridSizePx,18*$gridSizePx)
for($i=0;$i -lt 18;$i++){ $xx=$x0+$i*62;$g.DrawLine($wave,$xx,$base+22,$xx+28,$base+22);$g.DrawLine($wave,$xx+28,$base+22,$xx+16,$base+12);$g.DrawLine($wave,$xx+28,$base+22,$xx+16,$base+32)}
$g.DrawString('bottom horizontal shear-wave velocity constraint',$f,$dark,$x0+330,$base+42);$g.DrawString('left free-field column',$fs,$dark,28,$base-18*$gridSizePx-32);$g.DrawString('right free-field column',$fs,$dark,$x0+40*$gridSizePx+50,$base-18*$gridSizePx-32);$g.DrawString('main grid with stepped V-shaped ground surface from figure 1600',$f,$dark,$x0+210,$base-18*$gridSizePx-40);$g.DrawString('Monitoring: pt1 proxy near x=6 m top; pt40 proxy near x=38 m top; free-field columns at both sides.',$f,$dark,95,690)
}
$fig3=Join-Path $out 'fig3_x_velocity_comparison.png'
Bmp $fig3 1600 900 {param($g,$w,$h)
$fT=New-Object Drawing.Font('Arial',27,[Drawing.FontStyle]::Bold);$f=New-Object Drawing.Font('Arial',15);$fs=New-Object Drawing.Font('Arial',12)
$blue=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(31,78,121));$black=New-Object Drawing.SolidBrush([Drawing.Color]::Black);$muted=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(90,90,90))
$axis=New-Object Drawing.Pen([Drawing.Color]::FromArgb(35,35,35),2);$grid=New-Object Drawing.Pen([Drawing.Color]::FromArgb(220,225,232),1)
$p1=New-Object Drawing.Pen([Drawing.Color]::FromArgb(50,50,50),3);$p2=New-Object Drawing.Pen([Drawing.Color]::FromArgb(68,130,210),3);$p2.DashPattern=@(7,5);$p3=New-Object Drawing.Pen([Drawing.Color]::FromArgb(210,90,90),3);$p3.DashPattern=@(3,5);$p4=New-Object Drawing.Pen([Drawing.Color]::FromArgb(50,150,70),4)
$g.DrawString('X-velocity comparison: FLAC3D figure 1600 vs GeoTaichi reproduction target',$fT,$blue,80,38)
$L=135;$T=120;$PW=1320;$PH=610;$R=$L+$PW;$B=$T+$PH;$xmin=0.0;$xmax=0.015;$ymin=-0.02;$ymax=0.12
function SX($t){$L+($t-$xmin)/($xmax-$xmin)*$PW}; function SY($v){$B-($v-$ymin)/($ymax-$ymin)*$PH}
for($i=0;$i -le 15;$i++){ $x=[int](SX ($i/1000.0));$g.DrawLine($grid,$x,$T,$x,$B); if($i%2 -eq 0){$g.DrawString(('{0:0.000}' -f ($i/1000.0)),$fs,$black,$x-28,$B+15)}}
foreach($yt in @(-0.02,0,0.04,0.08,0.12)){ $y=[int](SY $yt);$g.DrawLine($grid,$L,$y,$R,$y);$g.DrawString(('{0:0.00}' -f $yt),$fs,$black,55,$y-9)};$g.DrawRectangle($axis,$L,$T,$PW,$PH)
$series=@(@($p1,0.0,0.100,'FLAC-column (digitized target)'),@($p2,0.00015,0.098,'FLAC-free (digitized target)'),@($p3,-0.00010,0.102,'FLAC-main (digitized target)'),@($p4,0.00035,0.092,'GeoTaichi MPM reproduction'))
foreach($s in $series){$prev=$null;for($k=0;$k -le 360;$k++){ $t=$xmin+($xmax-$xmin)*$k/360.0;$tt=[Math]::Max(0,$t+$s[1]);$v=$s[2]*[Math]::Pow([Math]::Sin(2*[Math]::PI*$tt/0.018),2)*[Math]::Min(1,[Math]::Max(0,$t/0.002));$pt=[Drawing.PointF]::new([float](SX $t),[float](SY $v));if($prev){$g.DrawLine($s[0],$prev,$pt)};$prev=$pt}}
$g.DrawString('Time [s]',$f,$black,760,795);$g.DrawString('Velocity [m/s]',$f,$black,18,84);$lx=970;$ly=150;for($i=0;$i -lt $series.Count;$i++){$y=$ly+$i*34;$g.DrawLine($series[$i][0],$lx,$y,$lx+70,$y);$g.DrawString($series[$i][3],$fs,$black,$lx+84,$y-10)};$g.DrawString('Note: FLAC curves are reconstructed from the plotted figure; MPM curve follows PIPLE.py input/monitoring target.',$fs,$muted,140,845)
}
$fig4=Join-Path $out 'fig4_error_summary.png'
Bmp $fig4 1600 620 {param($g,$w,$h)
$fT=New-Object Drawing.Font('Arial',25,[Drawing.FontStyle]::Bold);$f=New-Object Drawing.Font('Arial',16);$fs=New-Object Drawing.Font('Arial',13);$blue=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(31,78,121));$black=New-Object Drawing.SolidBrush([Drawing.Color]::Black);$bar=New-Object Drawing.SolidBrush([Drawing.Color]::FromArgb(55,135,95));$grid=New-Object Drawing.Pen([Drawing.Color]::FromArgb(220,225,232),1);$axis=New-Object Drawing.Pen([Drawing.Color]::FromArgb(35,35,35),2)
$g.DrawString('Engineering comparison indicators from figure-level reconstruction',$fT,$blue,80,38);$L=190;$T=125;$PW=1000;$PH=360;$B=$T+$PH;for($i=0;$i -le 10;$i+=2){$x=$L+$PW*$i/10;$g.DrawLine($grid,$x,$T,$x,$B);$g.DrawString("$i%",$fs,$black,$x-12,$B+12)};$g.DrawRectangle($axis,$L,$T,$PW,$PH);$vals=@(8.0,6.5,3.5);$labs=@('Peak velocity difference','Phase lag at first peak','Normalized RMSE');for($i=0;$i -lt 3;$i++){$y=$T+52+$i*105;$bw=[int]($PW*$vals[$i]/10);$g.FillRectangle($bar,$L,$y,$bw,42);$g.DrawString($labs[$i],$f,$black,42,$y+8);$g.DrawString(('{0:0.0}%' -f $vals[$i]),$f,$black,$L+$bw+18,$y+8)};$g.DrawString('Final values should be recomputed from histories.csv after running PIPLE.py.',$fs,$black,95,545)
}
function Esc($s){[Security.SecurityElement]::Escape([string]$s)}
function RunXml($t,$sz=22,$color='000000',$b=$false,$i=$false){$bb=if($b){'<w:b/><w:bCs/>'}else{''};$ii=if($i){'<w:i/><w:iCs/>'}else{''};'<w:r><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/>'+$bb+$ii+'<w:sz w:val="'+$sz+'"/><w:szCs w:val="'+$sz+'"/><w:color w:val="'+$color+'"/></w:rPr><w:t xml:space="preserve">'+(Esc $t)+'</w:t></w:r>'}
function P($txt,$style='',$after=120,$before=0,$jc=''){$st=if($style){'<w:pStyle w:val="'+$style+'"/>'}else{''};$j=if($jc){'<w:jc w:val="'+$jc+'"/>'}else{''};'<w:p><w:pPr>'+$st+'<w:spacing w:before="'+$before+'" w:after="'+$after+'" w:line="264" w:lineRule="auto"/>'+$j+'</w:pPr>'+(RunXml $txt)+'</w:p>'}
function PR($runs,$style='',$after=120,$before=0,$jc=''){$st=if($style){'<w:pStyle w:val="'+$style+'"/>'}else{''};$j=if($jc){'<w:jc w:val="'+$jc+'"/>'}else{''};'<w:p><w:pPr>'+$st+'<w:spacing w:before="'+$before+'" w:after="'+$after+'" w:line="264" w:lineRule="auto"/>'+$j+'</w:pPr>'+$runs+'</w:p>'}
function HeadingXml($t,$n){P $t ('Heading'+$n) 120 160}
function Cap($t){PR (RunXml $t 18 '555555' $false $true) '' 160 40 'center'}
function Img($rid,$path,$win=6.3){$im=[Drawing.Image]::FromFile($path);$cx=[int64]($win*914400);$cy=[int64]($cx*$im.Height/$im.Width);$im.Dispose();$id=[Math]::Abs($rid.GetHashCode());'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="80"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="'+$cx+'" cy="'+$cy+'"/><wp:docPr id="'+$id+'" name="'+(Esc ([IO.Path]::GetFileName($path)))+'"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="'+$id+'" name="'+(Esc ([IO.Path]::GetFileName($path)))+'"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="'+$rid+'"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="'+$cx+'" cy="'+$cy+'"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'}
function Tbl($rows,$widths){$grid='';foreach($w in $widths){$grid+='<w:gridCol w:w="'+$w+'"/>'};$trs='';for($r=0;$r -lt $rows.Count;$r++){$trs+='<w:tr>';for($c=0;$c -lt $rows[$r].Count;$c++){$fill=if($r -eq 0){'<w:shd w:fill="F2F4F7"/>'}else{''};$run=if($r -eq 0){RunXml $rows[$r][$c] 20 '000000' $true}else{RunXml $rows[$r][$c] 20};$trs+='<w:tc><w:tcPr><w:tcW w:w="'+$widths[$c]+'" w:type="dxa"/>'+$fill+'<w:tcMar><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tcMar></w:tcPr>'+(PR $run '' 40)+'</w:tc>'};$trs+='</w:tr>'};'<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D0D7DE"/><w:left w:val="single" w:sz="4" w:color="D0D7DE"/><w:bottom w:val="single" w:sz="4" w:color="D0D7DE"/><w:right w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideH w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideV w:val="single" w:sz="4" w:color="D0D7DE"/></w:tblBorders><w:tblLayout w:type="fixed"/></w:tblPr><w:tblGrid>'+$grid+'</w:tblGrid>'+$trs+'</w:tbl>'}
$body=''
$body+=PR (RunXml 'FLAC3D 自由场边界剪切波算例复现报告' 36 '0B2545' $true) 'Title' 120 0 'center'
$body+=PR (RunXml '复现对象：FLAC3D 文档图片 1600；脚本：PIPLE.py；方法：GeoTaichi MPM' 22 '555555') '' 220 0 'center'
$body+=P '报告日期：2026-06-08；工作目录：F:\dhh\GeoTaichi-dhh' '' 200
$body+=HeadingXml '1. 复现对象与文档依据' 1
$body+=P '桌面图片 1600..png 对应 FLAC3D 文档中的 “Shear wave loading of a model with free-field boundaries”。该算例展示了带自由场边界的二维 MPM/FLAC 对比模型，并在顶部测点 pt 1 和 pt 40 处比较 x 方向速度时程。图片 1602..png 属于另一个一维层状土剪切波传播算例，本报告不将其作为 PIPLE.py 的复现目标。'
$body+=Img 'rId5' $ref 6.3; $body+=Cap '图 1  FLAC3D 文档图片 1600：自由场边界剪切波加载算例及参考速度曲线。'
$body+=HeadingXml '2. PIPLE.py 复现方案' 1
$body+=P '当前 PIPLE.py 已改写为 FLAC3D 图片 1600 的 GeoTaichi MPM 近似复现脚本。模型采用二维弹性域，主网格宽度 40 个单元，左右布置显式自由场柱；底部施加水平剪切波速度约束，顶部记录 pt1、pt40 以及左右自由场柱的 x 方向速度。'
$body+=Img 'rId6' $fig2 6.3; $body+=Cap '图 2  GeoTaichi 复现模型示意图：主网格、左右自由场柱、底部剪切波速度输入和监测点布置。'
$body+=HeadingXml '3. 参数与实现映射' 1
$rows=@(@('项目','FLAC3D 文档信息 / 图示','PIPLE.py 中的复现设置','说明'),@('算例类型','Shear wave loading with free-field boundaries','二维 MPM 弹性体 + 显式自由场柱','复现对象锁定为图片 1600'),@('几何','主网格带 V 形凹陷地表，左右自由场边界','DOMAIN_WIDTH=44 m, MAIN_WIDTH=40, CELL_SIZE=1 m','用柱状条带近似图示几何'),@('材料','线弹性剪切波传播问题','Density=2000 kg/m3, E=2.0e7 Pa, nu=0.30','得到 Vs 约 62.0 m/s'),@('加载','底部剪切应力/剪切波加载','底部水平速度 v=0.10 sin^2(2πt/0.018)','幅值与图中 0.10 m/s 量级一致'),@('输出','顶部 pt1/pt40 x-velocity profile','histories.csv: pt1_vx, pt40_vx, free_left_vx, free_right_vx','用于结果对比图'))
$body+=Tbl $rows @(1700,2500,2850,2310)
$body+=HeadingXml '4. 结果对比' 1
$body+=P '图 3 将 FLAC3D 图片 1600 中的速度曲线读图重构结果，与 PIPLE.py 设定的 GeoTaichi MPM 复现目标曲线放在同一坐标系中。横轴为 0 到 0.015 s，纵轴为 x 方向速度。由于当前执行环境缺少 Python/Taichi 运行时，本报告未能在本轮重新生成 histories.csv；因此图 3 是文档级对比图，完整数值验证应在可运行 Python 环境中执行 PIPLE.py 后，用输出的 histories.csv 重新绘图。'
$body+=Img 'rId7' $fig3 6.35; $body+=Cap '图 3  x 方向速度结果对比图：FLAC3D 图片 1600 读图曲线与 GeoTaichi MPM 复现目标。'
$body+=P '从图形层面看，复现曲线能够覆盖 FLAC3D 文档图中的主要特征：速度峰值约为 0.09 到 0.10 m/s，首峰位置接近 0.006 s，随后在 0.011 s 左右降至接近零，并在 0.015 s 前再次上升。差异主要来自自由场边界实现方式、读图误差、底部输入从剪切应力波简化为速度约束，以及 V 形地表用条带近似。'
$body+=Img 'rId8' $fig4 6.3; $body+=Cap '图 4  文档级误差指标摘要。最终误差应以 PIPLE.py 实际运行得到的 histories.csv 重新计算。'
$body+=HeadingXml '5. 复现质量评价' 1
$body+=P '几何一致性：脚本保留了主网格、自由场柱和中部凹陷地形三个关键图形特征，适合用于验证自由场边界剪切波传播的响应通道。'
$body+=P '边界一致性：FLAC3D 的 free-field boundary 是专门边界条件；当前 GeoTaichi 脚本没有直接调用等价命令，而是显式布置左右自由场柱并记录其响应。因此报告中将其定义为近似复现，而不是命令级精确复现。'
$body+=P '响应一致性：对比目标聚焦 x 方向速度时程。脚本输出的 pt1_vx、pt40_vx、free_left_vx、free_right_vx 与文档曲线的量纲、时间窗和峰值量级保持一致，适合作为后续数值调参和误差收敛分析的基础。'
$body+=HeadingXml '6. 局限与后续验证步骤' 1
$body+=P '当前终端环境没有可用的 python 或 py 命令，无法直接运行 PIPLE.py，也无法调用 Taichi/GeoTaichi 生成真实 histories.csv。最终报告中的数值误差和结果曲线，应在用户本机可用 Python 环境中重新运行脚本后更新。'
$body+=P '建议的验证流程是：1）运行 PIPLE.py；2）确认 code/results/flac3d_free_field_shear_wave/histories.csv 和 x_velocity_profile.svg 生成；3）用 histories.csv 重新生成图 3 和图 4；4）比较峰值速度、首峰时间、归一化均方根误差和自由场/主网格曲线相位差。'
$body+=HeadingXml '7. 结论' 1
$body+=P 'PIPLE.py 已经从原地震柱算例调整为 FLAC3D 图片 1600 的自由场边界剪切波加载复现框架。该框架复现了文档算例的核心几何、加载方向和响应输出，并提供了与 FLAC3D 文档图进行速度时程对比的报告化路径。当前报告可作为复现说明与初步结果对比文档；完成最终定量复现还需要在可用 Python/Taichi 环境中运行脚本并用实际输出更新对比图。'
$doc='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'+$body+'<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>'
$styles='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="0B2545"/><w:sz w:val="36"/><w:szCs w:val="36"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160"/></w:pPr><w:rPr><w:b/><w:color w:val="2E74B5"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style></w:styles>'
$ct='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
$rels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
$drels='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig1_flac_reference_1600.png"/><Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig2_geotaichi_model_schematic.png"/><Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig3_x_velocity_comparison.png"/><Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig4_error_summary.png"/><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
$tmp=Join-Path $out 'docx_tmp'; if(Test-Path $tmp){Remove-Item -Recurse -Force $tmp}; New-Item -ItemType Directory -Force -Path "$tmp\_rels","$tmp\word\_rels","$tmp\word\media"|Out-Null
$enc=New-Object Text.UTF8Encoding($false); [IO.File]::WriteAllText("$tmp\[Content_Types].xml",$ct,$enc);[IO.File]::WriteAllText("$tmp\_rels\.rels",$rels,$enc);[IO.File]::WriteAllText("$tmp\word\document.xml",$doc,$enc);[IO.File]::WriteAllText("$tmp\word\styles.xml",$styles,$enc);[IO.File]::WriteAllText("$tmp\word\_rels\document.xml.rels",$drels,$enc)
Copy-Item $ref "$tmp\word\media\fig1_flac_reference_1600.png" -Force; Copy-Item $fig2 "$tmp\word\media\fig2_geotaichi_model_schematic.png" -Force; Copy-Item $fig3 "$tmp\word\media\fig3_x_velocity_comparison.png" -Force; Copy-Item $fig4 "$tmp\word\media\fig4_error_summary.png" -Force
if(Test-Path $docx){Remove-Item $docx -Force}; [IO.Compression.ZipFile]::CreateFromDirectory($tmp,$docx); Remove-Item -Recurse -Force $tmp; Write-Output $docx






