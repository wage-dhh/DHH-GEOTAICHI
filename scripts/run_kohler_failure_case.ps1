param(
    [Parameter(Mandatory = $true)]
    [string]$ConfigPath,
    [double[]]$PlotTimes = @(0, 1, 2, 3, 4),
    [switch]$UseStaticCheckpoint,
    [string]$StaticCheckpointPath = "output/kohler_static_convergence_4s/static_checkpoint.npz"
)

$ErrorActionPreference = "Continue"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configFile = (Resolve-Path (Join-Path $repoRoot $ConfigPath)).Path
$config = Get-Content $configFile -Raw | ConvertFrom-Json
$runDir = Join-Path $repoRoot $config.output_dir
$driver = Join-Path $repoRoot "example\mpm\NairnValidation\kohler_slope_seismic_mpm.py"
$plotter = Join-Path $repoRoot "scripts\plot_kohler_failure_snapshots.py"

Set-Location $repoRoot
$env:PYTHONUNBUFFERED = "1"
$env:NAIRN_SHEAR_OUTPUT = $runDir
$env:NAIRN_SHEAR_ARCH = "gpu"
$env:NAIRN_SHEAR_DX = [string]$config.dx
$env:NAIRN_SHEAR_DT = [string]$config.dt
$env:NAIRN_SHEAR_TIME = [string]$config.simulation_time
$env:NAIRN_SHEAR_SAVE_INTERVAL = [string]$config.save_interval
$env:NAIRN_SHEAR_HISTORY_INTERVAL = [string]$config.history_interval
$env:NAIRN_SHEAR_MAPPING = "USL"
$env:NAIRN_SHEAR_SHAPE = "CubicBSpline"
$env:NAIRN_SHEAR_STRICT_LOOP = "1"
$env:NAIRN_SHEAR_POSTPROCESS = "1"
$env:NAIRN_SHEAR_SAVE_PARTICLE = "1"
$env:NAIRN_SHEAR_SAVE_GRID = "0"
$env:NAIRN_RUN_DYNAMIC = "1"
$env:NAIRN_STATIC_CONVERGENCE_ONLY = "0"
$env:NAIRN_STATIC_CHECKPOINT_LOAD = if ($UseStaticCheckpoint) { "1" } else { "0" }
$env:NAIRN_STATIC_CHECKPOINT_CONTINUE = "0"
if ($UseStaticCheckpoint) {
    $env:NAIRN_STATIC_CHECKPOINT_PATH = (Resolve-Path (Join-Path $repoRoot $StaticCheckpointPath)).Path
    Remove-Item Env:NAIRN_STATIC_CHECKPOINT_SAVE_PATH -ErrorAction SilentlyContinue
} else {
    Remove-Item Env:NAIRN_STATIC_CHECKPOINT_PATH -ErrorAction SilentlyContinue
    $checkpointSavePath = Join-Path $runDir "static_checkpoint.npz"
    if ($null -ne $config.static_initialization.checkpoint_path) {
        $checkpointSavePath = Join-Path $repoRoot ([string]$config.static_initialization.checkpoint_path)
    }
    $env:NAIRN_STATIC_CHECKPOINT_SAVE_PATH = $checkpointSavePath
}
$env:NAIRN_STATIC_TIME = [string]$config.static_initialization.time
$env:NAIRN_STATIC_DT = [string]$config.static_initialization.dt
$env:NAIRN_STATIC_RAMP_TIME = [string]$config.static_initialization.ramp_time
$env:NAIRN_STATIC_SAVE_INTERVAL = [string]$config.static_initialization.save_interval
$env:NAIRN_STATIC_HISTORY_INTERVAL = [string]$config.static_initialization.history_interval
$env:NAIRN_STATIC_DAMPING = [string]$config.static_initialization.background_damping
$env:NAIRN_REQUIRE_STATIC_CONVERGENCE = if ([bool]$config.static_initialization.require_convergence) { "1" } else { "0" }
$env:NAIRN_STATIC_ALPHA_PIC = "1.0"
$env:NAIRN_MATERIAL_MODEL = [string]$config.material_model
$env:NAIRN_SOFTENING_ENABLED = "1"
# Strength values are defined in the JSON case.  Clear legacy environment
# overrides so the high-strength elastic base is not accidentally replaced by
# the soil strength.
foreach ($name in @(
    "NAIRN_DP_COHESION", "NAIRN_DP_FRICTION", "NAIRN_DP_DILATION", "NAIRN_DP_TENSILE", "NAIRN_DP_TYPE",
    "NAIRN_SOFTENING_RESIDUAL_COHESION", "NAIRN_SOFTENING_RESIDUAL_FRICTION", "NAIRN_SOFTENING_RESIDUAL_DILATION",
    "NAIRN_SOFTENING_EPS_START", "NAIRN_SOFTENING_EPS_END"
)) {
    Remove-Item ("Env:" + $name) -ErrorAction SilentlyContinue
}
$env:NAIRN_FREE_FIELD_GAP = "1.0"
$env:NAIRN_DYNAMIC_VELOCITY_PROJECTION = "Affine"
$env:NAIRN_DYNAMIC_ALPHA_PIC = "1.0"
$env:NAIRN_HUGHES_WINGET_STRESS_UPDATE = "1"
$env:NAIRN_INCREMENTAL_HISTORY = "1"
$env:NAIRN_INCREMENTAL_HISTORY_INTERVAL = [string]$config.history_interval
$boundaryDiagnosticInterval = 0.0
if ($null -ne $config.diagnostics -and $null -ne $config.diagnostics.boundary_diagnostic_interval) {
    $boundaryDiagnosticInterval = [double]$config.diagnostics.boundary_diagnostic_interval
}
$env:NAIRN_BOUNDARY_DIAGNOSTIC_INTERVAL = [string]$boundaryDiagnosticInterval
$env:NAIRN_INCREMENTAL_HISTORY_FSYNC = "1"
$domainEscapeDiagnostic = $true
$abortOnDomainEscape = $true
if ($null -ne $config.diagnostics) {
    if ($null -ne $config.diagnostics.domain_escape) {
        $domainEscapeDiagnostic = [bool]$config.diagnostics.domain_escape
    }
    if ($null -ne $config.diagnostics.abort_on_domain_escape) {
        $abortOnDomainEscape = [bool]$config.diagnostics.abort_on_domain_escape
    }
}
$env:NAIRN_DOMAIN_ESCAPE_DIAGNOSTIC = if ($domainEscapeDiagnostic) { "1" } else { "0" }
$env:NAIRN_ABORT_ON_DOMAIN_ESCAPE = if ($abortOnDomainEscape) { "1" } else { "0" }
$env:NAIRN_SEISMIC_INPUT_MODE = [string]$config.earthquake_input.mode
$env:NAIRN_SEISMIC_INPUT_FACTOR = [string]$config.earthquake_input.input_velocity_factor
$env:NAIRN_EARTHQUAKE_ENABLED = if ([bool]$config.earthquake_input.enabled) { "1" } else { "0" }
if ($config.earthquake_input.mode -eq "AT2") {
    $env:NAIRN_EARTHQUAKE_FILE = [string]$config.earthquake_input.file
} else {
    $env:NAIRN_FIG10C_INPUT_FILE = [string]$config.earthquake_input.fig10c_file
}

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
& py -3 $driver --config $configFile
$solverExitCode = $LASTEXITCODE

$dynamicStart = [double]$config.static_initialization.time
$dynamicStartSource = "config.static_initialization.time (fallback)"
$staticReport = Join-Path $runDir "static_initialization_report.md"
if (Test-Path -LiteralPath $staticReport) {
    $dynamicStartLine = Select-String -Path $staticReport -Pattern "dynamic_stage_starts_at:" | Select-Object -First 1
    if ($null -ne $dynamicStartLine -and $dynamicStartLine.Line -match 'dynamic_stage_starts_at:\s*`([^`]+)`') {
        try {
            $dynamicStart = [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture)
            $dynamicStartSource = "static_initialization_report.md"
        } catch {
            Write-Warning "Could not parse dynamic_stage_starts_at from $staticReport; using config fallback."
        }
    }
}
$plotArgs = @($plotter, $runDir, "--dynamic-start", $dynamicStart, "--times") + @($PlotTimes | ForEach-Object { [string]$_ }) + @(
    "--output", (Join-Path $runDir "kohler_failure_snapshots.png"),
    "--summary", (Join-Path $runDir "kohler_failure_snapshots.csv")
)
& py -3 $plotArgs
$plotExitCode = $LASTEXITCODE

@(
    "config=$configFile"
    "run_dir=$runDir"
    "solver_exit_code=$solverExitCode"
    "plot_exit_code=$plotExitCode"
    "dynamic_start=$dynamicStart"
    "dynamic_start_source=$dynamicStartSource"
    "finish_time=$(Get-Date -Format o)"
) | Set-Content (Join-Path $runDir "run_status.txt") -Encoding UTF8

if ($solverExitCode -ne 0) { exit $solverExitCode }
exit $plotExitCode
