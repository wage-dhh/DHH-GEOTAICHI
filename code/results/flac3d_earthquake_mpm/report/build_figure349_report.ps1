$ErrorActionPreference = 'Stop'

$root = 'F:\dhh\GeoTaichi-dhh'
$resultDir = Join-Path $root 'code\results\flac3d_earthquake_mpm'
$reportDir = Join-Path $resultDir 'report'
$docxPath = Join-Path $reportDir 'FLAC3D_Figure_3_49_MPM_reproduction_report.docx'
$workDir = Join-Path $reportDir '_figure349_docx_work'

if (Test-Path $workDir) { Remove-Item -LiteralPath $workDir -Recurse -Force }
New-Item -ItemType Directory -Path $workDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $workDir '_rels') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $workDir 'word\_rels') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $workDir 'word\media') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $workDir 'docProps') | Out-Null

$summaryPath = Join-Path $resultDir 'validation_summary.txt'
$summaryText = if (Test-Path $summaryPath) { Get-Content -LiteralPath $summaryPath -Raw -Encoding UTF8 } else { '' }
$summary = @{}
foreach ($line in ($summaryText -split "`r?`n")) {
    if ($line -match '^\s*([^=]+?)\s*=\s*(.*)\s*$') {
        $summary[$matches[1].Trim()] = $matches[2].Trim()
    }
}

function SummaryValue([string]$key, [string]$fallback) {
    if ($summary.ContainsKey($key) -and $summary[$key]) {
        return [string]$summary[$key]
    }
    return $fallback
}

$images = @(
    @{ Path = (Join-Path $reportDir 'fig1_input_wave.png'); Name = 'Figure 1  Function input acceleration corresponding to FLAC3D Figure 3.49.' },
    @{ Path = (Join-Path $reportDir 'fig2_acceleration_comparison.png'); Name = 'Figure 2  Acceleration histories: MPM response compared with document result ranges.' },
    @{ Path = (Join-Path $reportDir 'fig3_stress_strain_comparison.png'); Name = 'Figure 3  Shear stress-strain paths at base and middle monitoring zones.' },
    @{ Path = (Join-Path $reportDir 'fig4_displacement.png'); Name = 'Figure 4  Horizontal displacement histories of base, middle, and surface zones.' }
)

Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.IO.Compression.FileSystem

function X([object]$s) {
    return [System.Security.SecurityElement]::Escape([string]$s)
}

function RunXml([string]$text, [int]$size = 22, [string]$color = '000000', [bool]$bold = $false, [bool]$italic = $false) {
    $flags = ''
    if ($bold) { $flags += '<w:b/><w:bCs/>' }
    if ($italic) { $flags += '<w:i/><w:iCs/>' }
    return '<w:r><w:rPr>' + $flags + '<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="' + $size + '"/><w:szCs w:val="' + $size + '"/><w:color w:val="' + $color + '"/></w:rPr><w:t xml:space="preserve">' + (X $text) + '</w:t></w:r>'
}

function ParaXml([string]$text, [string]$style = '', [int]$before = 0, [int]$after = 120, [int]$line = 264, [string]$jc = '', [string]$runs = '') {
    $ppr = ''
    if ($style) { $ppr += '<w:pStyle w:val="' + $style + '"/>' }
    $ppr += '<w:spacing w:before="' + $before + '" w:after="' + $after + '" w:line="' + $line + '" w:lineRule="auto"/>'
    if ($jc) { $ppr += '<w:jc w:val="' + $jc + '"/>' }
    if (-not $runs) { $runs = RunXml $text }
    return '<w:p><w:pPr>' + $ppr + '</w:pPr>' + $runs + '</w:p>'
}

function HeadingXml([string]$text, [int]$level) {
    if ($level -eq 1) { return ParaXml $text 'Heading1' 320 160 264 }
    if ($level -eq 2) { return ParaXml $text 'Heading2' 240 120 264 }
    return ParaXml $text 'Heading3' 160 80 264
}

function CaptionXml([string]$text) {
    return ParaXml '' '' 40 180 250 'center' (RunXml $text 18 '555555' $false $true)
}

function PageBreakXml() {
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
}

function CellXml([string]$text, [int]$width, [bool]$header = $false) {
    $fill = if ($header) { '<w:shd w:fill="F2F4F7"/>' } else { '' }
    $runs = if ($header) { RunXml $text 21 '0B2545' $true $false } else { RunXml $text 20 '000000' $false $false }
    $p = ParaXml '' '' 0 40 260 '' $runs
    return '<w:tc><w:tcPr><w:tcW w:w="' + $width + '" w:type="dxa"/><w:vAlign w:val="center"/>' + $fill + '<w:tcMar><w:top w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:start w:w="140" w:type="dxa"/><w:end w:w="140" w:type="dxa"/></w:tcMar></w:tcPr>' + $p + '</w:tc>'
}

function TableXml([object[]]$rows, [int[]]$widths) {
    $grid = ''
    foreach ($w in $widths) { $grid += '<w:gridCol w:w="' + $w + '"/>' }
    $trs = ''
    for ($i = 0; $i -lt $rows.Count; $i++) {
        $header = ($i -eq 0)
        $cells = ''
        for ($j = 0; $j -lt $rows[$i].Count; $j++) {
            $cells += CellXml ([string]$rows[$i][$j]) $widths[$j] $header
        }
        $trs += '<w:tr>' + $cells + '</w:tr>'
    }
    return '<w:tbl><w:tblPr><w:tblW w:w="9360" w:type="dxa"/><w:tblInd w:w="120" w:type="dxa"/><w:tblBorders><w:top w:val="single" w:sz="4" w:color="D0D7DE"/><w:left w:val="single" w:sz="4" w:color="D0D7DE"/><w:bottom w:val="single" w:sz="4" w:color="D0D7DE"/><w:right w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideH w:val="single" w:sz="4" w:color="D0D7DE"/><w:insideV w:val="single" w:sz="4" w:color="D0D7DE"/></w:tblBorders><w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:start w:w="120" w:type="dxa"/><w:end w:w="120" w:type="dxa"/></w:tblCellMar></w:tblPr><w:tblGrid>' + $grid + '</w:tblGrid>' + $trs + '</w:tbl>'
}

$imageRels = New-Object System.Collections.Generic.List[object]
function ImageXml([string]$path, [double]$widthIn = 6.25) {
    $idx = $script:imageRels.Count + 1
    $rid = 'rIdImg' + $idx
    $mediaName = 'image' + $idx + '.png'
    Copy-Item -LiteralPath $path -Destination (Join-Path $workDir ('word\media\' + $mediaName)) -Force
    $script:imageRels.Add(@{ Rid = $rid; Target = 'media/' + $mediaName }) | Out-Null
    $img = [System.Drawing.Image]::FromFile($path)
    try {
        $cx = [int64]($widthIn * 914400)
        $cy = [int64]($cx * $img.Height / $img.Width)
    } finally {
        $img.Dispose()
    }
    return '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="80" w:after="80"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="' + $cx + '" cy="' + $cy + '"/><wp:docPr id="' + $idx + '" name="result figure"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="' + $idx + '" name="result figure"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="' + $rid + '"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="' + $cx + '" cy="' + $cy + '"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
}

$body = ''
$body += ParaXml '' '' 0 80 264 'center' (RunXml 'FLAC3D Figure 3.49 地震输入算例的 GeoTaichi MPM 复现报告' 34 '0B2545' $true $false)
$body += ParaXml '' '' 0 260 264 'center' (RunXml '对象：桌面 lizhi.pdf / Figure 3.49 Input acceleration at bottom of model；实现脚本：PIPLE.py' 21 '555555' $false $false)
$body += '<w:p><w:pPr><w:pBdr><w:bottom w:val="single" w:sz="8" w:space="3" w:color="2E74B5"/></w:pBdr><w:spacing w:after="260"/></w:pPr></w:p>'

$body += HeadingXml '1. 复现目标' 1
$body += ParaXml '本报告说明 PIPLE.py 对 FLAC3D 文档中 Figure 3.49 “Input acceleration at bottom of model” 的复现方式。该图给出模型底部的水平输入加速度时程；在 GeoTaichi MPM 中，脚本将该加速度积分为底部水平速度约束，并输出底部、中部和自由表面的动力响应。'
$body += ParaXml '当前报告复用 code/results/flac3d_earthquake_mpm 中已有的计算结果图与 CSV 数据。由于本环境没有可用 python/py 命令，本轮没有重新运行 PIPLE.py；报告中的图形对应已有结果文件。'

$body += HeadingXml '2. 关键设置' 1
$rows = @(
    @('项目','数值 / 文件','说明'),
    @('源文档','C:\Users\Dell\Desktop\lizhi.pdf','Figure 3.49 底部输入加速度'),
    @('脚本','PIPLE.py','底部输入加速度积分为 x 方向速度约束'),
    @('输出目录','code/results/flac3d_earthquake_mpm','保存 CSV、SVG、PNG 和 DOCX 报告'),
    @('土层高度','20 m','二维一列剪切柱，宽度 1 m'),
    @('密度',(SummaryValue 'density' '1000.0') + ' kg/m3','文档算例参数'),
    @('剪切模量 G',(SummaryValue 'shear_modulus' '5.0e8') + ' Pa','用于剪切波传播和应力-应变换算'),
    @('体积模量 K',(SummaryValue 'bulk_modulus' '1.0e9') + ' Pa','换算 Young 模量和 Poisson 比'),
    @('时间步长',(SummaryValue 'dt' '2.0e-4') + ' s','PIPLE.py 默认时间步'),
    @('输入加速度范围',(SummaryValue 'input_acceleration_range' '[-8.656, 8.199]') + ' m/s2','按 Figure 3.49 / 文档结果范围标定')
)
$body += TableXml $rows @(2100,3100,4160)

$body += HeadingXml '3. 输入波形与响应对比' 1
$body += ParaXml '图 1 给出用于底部加载的函数型输入加速度。图 2 将底部、中部和自由表面三处加速度响应与文档图中的量级范围叠加比较。阴影区表示文档范围，曲线表示 MPM 输出。'
if (Test-Path $images[0].Path) { $body += ImageXml $images[0].Path 6.25; $body += CaptionXml $images[0].Name }
if (Test-Path $images[1].Path) { $body += ImageXml $images[1].Path 6.25; $body += CaptionXml $images[1].Name }

$body += HeadingXml '4. 应力-应变与位移检查' 1
$body += ParaXml '剪应力-剪应变图用于检查动力输入后土柱内部响应是否落在文档结果量级内；位移历史用于检查输入积分后的基底速度是否引入异常长期漂移。'
if (Test-Path $images[2].Path) { $body += ImageXml $images[2].Path 6.15; $body += CaptionXml $images[2].Name }
if (Test-Path $images[3].Path) { $body += ImageXml $images[3].Path 6.15; $body += CaptionXml $images[3].Name }

$body += HeadingXml '5. 复现结论与限制' 1
$body += ParaXml 'PIPLE.py 已将 Figure 3.49 的底部加速度输入转化为可重复的解析/CSV 驱动加载流程，并提供 base_motion.csv、histories.csv、acceleration_comparison.svg 和 stress_strain_comparison.svg 作为结果核查文件。'
$body += ParaXml '需要注意：当前输入波形是按文档图形范围和波形包络拟合/复用的函数型时程，而不是 FLAC3D 原始 gilroy1.acc 文件逐点记录。因此，该复现适合验证建模流程、响应量级和对比图生成，不应解释为与 FLAC3D 原始记录逐点完全一致。'

$body += HeadingXml '6. 文件清单' 1
$fileRows = @(
    @('文件','用途'),
    @('PIPLE.py','GeoTaichi MPM 复现脚本，包含 Figure 3.49 底部输入加速度处理'),
    @('base_motion.csv','输入加速度及积分底部速度'),
    @('histories.csv','底部、中部、自由表面速度、加速度、位移和应力历史'),
    @('fig1_input_wave.png','报告中的底部输入波形图'),
    @('fig2_acceleration_comparison.png','报告中的加速度响应对比图'),
    @('fig3_stress_strain_comparison.png','报告中的剪应力-剪应变对比图'),
    @('fig4_displacement.png','报告中的水平位移历史图')
)
$body += TableXml $fileRows @(3300,6060)

$sectPr = '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="708" w:footer="708" w:gutter="0"/><w:cols w:space="708"/><w:docGrid w:linePitch="360"/></w:sectPr>'

$document = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" mc:Ignorable="w14 w15 wp14"><w:body>' + $body + $sectPr + '</w:body></w:document>'

$styles = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="264" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="0" w:after="120" w:line="264" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="320" w:after="160" w:line="264" w:lineRule="auto"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:bCs/><w:color w:val="2E74B5"/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120" w:line="264" w:lineRule="auto"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:bCs/><w:color w:val="2E74B5"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style><w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="160" w:after="80" w:line="264" w:lineRule="auto"/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:eastAsia="Microsoft YaHei"/><w:b/><w:bCs/><w:color w:val="1F4D78"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style></w:styles>'

$rels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>'

$docRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
foreach ($rel in $imageRels) {
    $docRels += '<Relationship Id="' + $rel.Rid + '" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="' + $rel.Target + '"/>'
}
$docRels += '</Relationships>'

$contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>'

$now = [DateTime]::UtcNow.ToString('s') + 'Z'
$core = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>FLAC3D Figure 3.49 MPM reproduction report</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">' + $now + '</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">' + $now + '</dcterms:modified></cp:coreProperties>'
$app = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Codex</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company></Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>16.0000</AppVersion></Properties>'

Set-Content -LiteralPath (Join-Path $workDir '[Content_Types].xml') -Value $contentTypes -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir '_rels\.rels') -Value $rels -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir 'word\document.xml') -Value $document -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir 'word\styles.xml') -Value $styles -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir 'word\_rels\document.xml.rels') -Value $docRels -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir 'docProps\core.xml') -Value $core -Encoding UTF8
Set-Content -LiteralPath (Join-Path $workDir 'docProps\app.xml') -Value $app -Encoding UTF8

if (Test-Path $docxPath) { Remove-Item -LiteralPath $docxPath -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($workDir, $docxPath)

Write-Output $docxPath
