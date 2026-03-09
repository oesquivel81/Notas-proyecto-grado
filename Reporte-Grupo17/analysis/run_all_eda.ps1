param(
    [string]$DatasetRoot = "MaIA_Scoliosis_Dataset",
    [string]$PythonExe = "./.venv/Scripts/python.exe",
    [int]$Seed = 42,
    [double]$Train = 0.7,
    [double]$Val = 0.15,
    [double]$Test = 0.15
)

$ErrorActionPreference = "Stop"

Write-Host "[1/3] Running base EDA..." -ForegroundColor Cyan
& $PythonExe analysis/eda_maia_dataset.py `
    --dataset-root $DatasetRoot `
    --out-json analysis/eda_report.json

Write-Host "[2/3] Generating figures..." -ForegroundColor Cyan
& $PythonExe analysis/plot_maia_dataset.py `
    --dataset-root $DatasetRoot `
    --out-dir analysis/figures

Write-Host "[3/3] Creating patient-level split..." -ForegroundColor Cyan
& $PythonExe analysis/create_patient_split.py `
    --dataset-root $DatasetRoot `
    --seed $Seed `
    --train $Train `
    --val $Val `
    --test $Test `
    --out-csv analysis/dataset_index_patient_split.csv `
    --out-json analysis/patient_split_summary.json

Write-Host "Done. Outputs generated under analysis/." -ForegroundColor Green
