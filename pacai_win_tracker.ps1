# pacai_win_tracker.ps1

$redWins = 0
$blueWins = 0

for ($seed = 1; $seed -le 10; $seed++) {
    $output = python3 -m pacai.bin.capture --red pacai.student.galaxyTeam --blue pacai.core.baselineTeam --seed $seed 2>&1
    $output | ForEach-Object {
        if ($_ -match "The Red team wins") {
            $redWins++
        } elseif ($_ -match "The Blue team wins") {
            $blueWins++
        }
        Write-Output $_  # Display game log in real-time
    }
}

Write-Output "Final Results:"
Write-Output "Red Team Wins: $redWins"
Write-Output "Blue Team Wins: $blueWins"
