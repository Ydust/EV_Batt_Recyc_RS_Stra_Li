param(
    [int]$MaxJobs = 4
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BaseOut = Join-Path $Root "unified_policy_run\data\lithium_loss_scenarios_unified_yearly_parallel"
New-Item -ItemType Directory -Force -Path $BaseOut | Out-Null

$Years = 2025..2050
$CommonArgs = @(
    ".\run_lithium_loss_scenarios_unified.py",
    "--include-pyrohydro",
    "--pyrohydro-pyro-weight", "0.25",
    "--pyrohydro-group-multipliers", "developed=0.78,ev_producer=0.82,other=0.90",
    "--direct-cost-multiplier", "1.2",
    "--availability-threshold", "0.6",
    "--include-max-li"
)

function Receive-FinishedJobs {
    foreach ($job in Get-Job | Where-Object { $_.State -ne "Running" }) {
        Receive-Job $job
        if ($job.State -ne "Completed") {
            $name = $job.Name
            Remove-Job $job -Force
            throw "Job $name failed with state $($job.State)."
        }
        Remove-Job $job -Force
    }
}

foreach ($year in $Years) {
    $OutDir = Join-Path $BaseOut "year_$year"
    $Summary = Join-Path $OutDir "lithium_loss_scenarios_summary.csv"
    if (Test-Path $Summary) {
        Write-Host "[$(Get-Date -Format HH:mm:ss)] skip $year, summary exists"
        continue
    }

    while ((Get-Job | Where-Object { $_.State -eq "Running" }).Count -ge $MaxJobs) {
        Receive-FinishedJobs
        Start-Sleep -Seconds 10
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $LogPath = Join-Path $OutDir "run.log"
    Write-Host "[$(Get-Date -Format HH:mm:ss)] start $year"
    Start-Job -Name "fig4_$year" -ScriptBlock {
        param($Root, $Year, $OutDir, $CommonArgs, $LogPath)
        Set-Location $Root
        $args = @($CommonArgs + @("--years", [string]$Year, "--output-dir", $OutDir))
        & python @args *>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "python exited with code $LASTEXITCODE for year $Year"
        }
    } -ArgumentList $Root, $year, $OutDir, $CommonArgs, $LogPath | Out-Null
}

while ((Get-Job).Count -gt 0) {
    Receive-FinishedJobs
    Start-Sleep -Seconds 10
}

Write-Host "[$(Get-Date -Format HH:mm:ss)] all yearly Fig4 jobs completed"
