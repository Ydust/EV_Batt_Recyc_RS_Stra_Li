param(
    [int]$MaxJobs = 8,
    [string[]]$OnlyConfigs = @()
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$BaseOut = Join-Path $Root "unified_policy_run\data\lithium_loss_scenarios_yearly_sensitivity"
$PolicyOut = Join-Path $BaseOut "_policy_files"
New-Item -ItemType Directory -Force -Path $BaseOut | Out-Null
New-Item -ItemType Directory -Force -Path $PolicyOut | Out-Null

$BaselinePolicy = Join-Path $Root "Figure_data\joint_policy_technology\waste_trade_policy_constraints_reference_relaxed.csv"

function New-ChinaPenaltyPolicyFile {
    param(
        [int]$Penalty
    )
    $OutFile = Join-Path $PolicyOut "waste_trade_policy_constraints_china_penalty_$Penalty.csv"
    if (-not (Test-Path $OutFile)) {
        $rows = Import-Csv $BaselinePolicy
        foreach ($row in $rows) {
            if ($row.scenario -eq "current_policy" -and $row.destination_country -eq "China") {
                $row.policy_penalty_usd_per_t = [string]$Penalty
            }
        }
        $rows | Export-Csv -NoTypeInformation -Encoding UTF8 $OutFile
    }
    return $OutFile
}

$PolicyPenalty150 = New-ChinaPenaltyPolicyFile -Penalty 150
$PolicyPenalty600 = New-ChinaPenaltyPolicyFile -Penalty 600

$Configs = @(
    [pscustomobject]@{ Name="direct_mult_1.0"; DirectCost="1.0"; Availability="0.6"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="direct_mult_1.5"; DirectCost="1.5"; Availability="0.6"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="avail_thr_0.4"; DirectCost="1.2"; Availability="0.4"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="avail_thr_0.8"; DirectCost="1.2"; Availability="0.8"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="penalty_150"; DirectCost="1.2"; Availability="0.6"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$PolicyPenalty150 },
    [pscustomobject]@{ Name="penalty_600"; DirectCost="1.2"; Availability="0.6"; DelayCost="1.0"; LithiumPrice=$null; PolicyFile=$PolicyPenalty600 },
    [pscustomobject]@{ Name="delay_5"; DirectCost="1.2"; Availability="0.6"; DelayCost="5.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="delay_10"; DirectCost="1.2"; Availability="0.6"; DelayCost="10.0"; LithiumPrice=$null; PolicyFile=$null },
    [pscustomobject]@{ Name="li_price_3"; DirectCost="1.2"; Availability="0.6"; DelayCost="1.0"; LithiumPrice="3.0"; PolicyFile=$null },
    [pscustomobject]@{ Name="li_price_5"; DirectCost="1.2"; Availability="0.6"; DelayCost="1.0"; LithiumPrice="5.0"; PolicyFile=$null }
)

if ($OnlyConfigs.Count -gt 0) {
    $wanted = @{}
    foreach ($name in $OnlyConfigs) {
        $wanted[$name] = $true
    }
    $Configs = $Configs | Where-Object { $wanted.ContainsKey($_.Name) }
}

$CommonArgs = @(
    ".\run_lithium_loss_scenarios_unified.py",
    "--include-pyrohydro",
    "--pyrohydro-pyro-weight", "0.25",
    "--pyrohydro-group-multipliers", "developed=0.78,ev_producer=0.82,other=0.90",
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

$Years = 2025..2050
foreach ($config in $Configs) {
    foreach ($year in $Years) {
        $OutDir = Join-Path $BaseOut "$($config.Name)\year_$year"
        $Summary = Join-Path $OutDir "lithium_loss_scenarios_summary.csv"
        if (Test-Path $Summary) {
            Write-Host "[$(Get-Date -Format HH:mm:ss)] skip $($config.Name) $year, summary exists"
            continue
        }

        while ((Get-Job | Where-Object { $_.State -eq "Running" }).Count -ge $MaxJobs) {
            Receive-FinishedJobs
            Start-Sleep -Seconds 10
        }

        New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
        $LogPath = Join-Path $OutDir "run.log"
        Write-Host "[$(Get-Date -Format HH:mm:ss)] start $($config.Name) $year"

        Start-Job -Name "fig4_$($config.Name)_$year" -ScriptBlock {
            param($Root, $Year, $OutDir, $LogPath, $CommonArgs, $DirectCost, $Availability, $DelayCost, $LithiumPrice, $PolicyFile)
            Set-Location $Root
            $args = @($CommonArgs + @(
                "--years", [string]$Year,
                "--direct-cost-multiplier", [string]$DirectCost,
                "--availability-threshold", [string]$Availability,
                "--delay-cost", [string]$DelayCost,
                "--output-dir", $OutDir
            ))
            if ($LithiumPrice) {
                $args += @("--lithium-price-multiplier", [string]$LithiumPrice)
            }
            if ($PolicyFile) {
                $args += @("--policy-file", [string]$PolicyFile)
            }
            & python @args *>&1 | Tee-Object -FilePath $LogPath
            if ($LASTEXITCODE -ne 0) {
                throw "python exited with code $LASTEXITCODE for $Year"
            }
        } -ArgumentList $Root, $year, $OutDir, $LogPath, $CommonArgs, $config.DirectCost, $config.Availability, $config.DelayCost, $config.LithiumPrice, $config.PolicyFile | Out-Null
    }
}

while ((Get-Job).Count -gt 0) {
    Receive-FinishedJobs
    Start-Sleep -Seconds 10
}

Write-Host "[$(Get-Date -Format HH:mm:ss)] all yearly Fig4 sensitivity jobs completed"
