param(
    [int]$MaxJobs = 4,
    [string[]]$OnlyCases = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$InputDir = Join-Path $Root "unified_policy_run\data\demand_side_inputs"
$BaseOut = Join-Path $Root "unified_policy_run\data\fig4_demand_side_sensitivity"
New-Item -ItemType Directory -Force -Path $InputDir | Out-Null
New-Item -ItemType Directory -Force -Path $BaseOut | Out-Null

if (-not (Test-Path (Join-Path $InputDir "eol_plus_manufacturing_collected_by_scenario.csv"))) {
    & python ".\build_unified_demand_side_inputs.py"
    if ($LASTEXITCODE -ne 0) {
        throw "build_unified_demand_side_inputs.py failed with code $LASTEXITCODE"
    }
}

$Cases = @(
    [pscustomobject]@{
        Name="eol_only"
        ScrapInput=(Join-Path $InputDir "eol_collected_by_scenario.csv")
    },
    [pscustomobject]@{
        Name="eol_plus_manufacturing"
        ScrapInput=(Join-Path $InputDir "eol_plus_manufacturing_collected_by_scenario.csv")
    }
)

if ($OnlyCases.Count -gt 0) {
    $wanted = @{}
    foreach ($name in $OnlyCases) {
        $wanted[$name] = $true
    }
    $Cases = $Cases | Where-Object { $wanted.ContainsKey($_.Name) }
}

$Collections = @("low_collection", "baseline", "high_collection")
$Recoveries = @("low", "baseline", "high")
$Years = "2030,2040,2050"

$CommonArgs = @(
    ".\run_lithium_loss_scenarios_unified.py",
    "--include-pyrohydro",
    "--pyrohydro-pyro-weight", "0.25",
    "--pyrohydro-group-multipliers", "developed=0.78,ev_producer=0.82,other=0.90",
    "--include-max-li",
    "--years", $Years
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

foreach ($case in $Cases) {
    foreach ($collection in $Collections) {
        foreach ($recovery in $Recoveries) {
            $OutDir = Join-Path $BaseOut "$($case.Name)\collection_$collection\recovery_$recovery"
            $Summary = Join-Path $OutDir "lithium_loss_scenarios_summary.csv"
            if (Test-Path $Summary) {
                Write-Host "[$(Get-Date -Format HH:mm:ss)] skip $($case.Name) $collection $recovery, summary exists"
                continue
            }

            while ((Get-Job | Where-Object { $_.State -eq "Running" }).Count -ge $MaxJobs) {
                Receive-FinishedJobs
                Start-Sleep -Seconds 10
            }

            New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
            $LogPath = Join-Path $OutDir "run.log"
            Write-Host "[$(Get-Date -Format HH:mm:ss)] start $($case.Name) $collection $recovery"

            Start-Job -Name "fig4_demand_$($case.Name)_$($collection)_$($recovery)" -ScriptBlock {
                param($Root, $CommonArgs, $ScrapInput, $Collection, $Recovery, $OutDir, $LogPath)
                Set-Location $Root
                $args = @($CommonArgs + @(
                    "--scrap-input-file", [string]$ScrapInput,
                    "--collection-scenario", [string]$Collection,
                    "--recovery-scenario", [string]$Recovery,
                    "--output-dir", [string]$OutDir
                ))
                & python @args *>&1 | Tee-Object -FilePath $LogPath
                if ($LASTEXITCODE -ne 0) {
                    throw "python exited with code $LASTEXITCODE for $Collection / $Recovery"
                }
            } -ArgumentList $Root, $CommonArgs, $case.ScrapInput, $collection, $recovery, $OutDir, $LogPath | Out-Null
        }
    }
}

while ((Get-Job).Count -gt 0) {
    Receive-FinishedJobs
    Start-Sleep -Seconds 10
}

Write-Host "[$(Get-Date -Format HH:mm:ss)] all Fig4 demand-side sensitivity jobs completed"
