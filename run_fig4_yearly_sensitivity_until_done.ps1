param(
    [int]$MaxJobs = 8,
    [int]$SleepSeconds = 60
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BaseOut = Join-Path $Root "unified_policy_run\data\lithium_loss_scenarios_yearly_sensitivity"
$LogDir = Join-Path $BaseOut "_runner_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunnerLog = Join-Path $LogDir "until_done.log"

$ExpectedConfigs = @(
    "direct_mult_1.0",
    "direct_mult_1.5",
    "avail_thr_0.4",
    "avail_thr_0.8",
    "penalty_150",
    "penalty_600",
    "delay_5",
    "delay_10",
    "li_price_3",
    "li_price_5"
)

function Write-RunnerLog {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    $line | Tee-Object -FilePath $RunnerLog -Append
}

function Get-ProgressRows {
    $rows = @()
    foreach ($config in $ExpectedConfigs) {
        $ConfigDir = Join-Path $BaseOut $config
        $files = @()
        if (Test-Path $ConfigDir) {
            $files = Get-ChildItem -Path (Join-Path $ConfigDir "year_*\lithium_loss_scenarios_summary.csv") -ErrorAction SilentlyContinue
        }
        $years = $files | ForEach-Object {
            [int]((Split-Path $_.DirectoryName -Leaf) -replace "year_", "")
        } | Sort-Object
        $missing = @(2025..2050 | Where-Object { $_ -notin $years })
        $rows += [pscustomobject]@{
            Config = $config
            Completed = $files.Count
            Missing = ($missing -join ",")
        }
    }
    return $rows
}

Write-RunnerLog "starting until-done runner with MaxJobs=$MaxJobs"

while ($true) {
    $progress = Get-ProgressRows
    $done = ($progress | Where-Object { $_.Completed -ge 26 }).Count
    $total = ($progress | Measure-Object Completed -Sum).Sum
    Write-RunnerLog "progress: $done/10 configs complete, $total/260 yearly jobs complete"
    $progress | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $RunnerLog -Append | Out-Null

    if ($done -eq $ExpectedConfigs.Count) {
        Write-RunnerLog "all yearly Fig4 sensitivity jobs completed"
        break
    }

    $RunLog = Join-Path $LogDir ("scheduler_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
    Write-RunnerLog "launching scheduler; log=$RunLog"
    try {
        & "$Root\run_fig4_yearly_sensitivity_parallel.ps1" -MaxJobs $MaxJobs *>&1 |
            Tee-Object -FilePath $RunLog
        Write-RunnerLog "scheduler exited with code $LASTEXITCODE"
    }
    catch {
        Write-RunnerLog "scheduler threw: $($_.Exception.Message)"
    }

    Start-Sleep -Seconds $SleepSeconds
}
