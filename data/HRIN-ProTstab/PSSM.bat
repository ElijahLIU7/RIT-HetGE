@echo off
REM Enable delayed variable expansion
setlocal enabledelayedexpansion

REM Define paths
set "query_folder=FASTA\train_dataset_fasta"
set "output_folder=PSSM\train_pssm"
set "db_path=blast-2.15.0+\swissprot\swissprot"
set "evalue=0.001"
set "iterations=3"

REM Create output folder if it doesn't exist
if not exist "%output_folder%" mkdir "%output_folder%"

REM Loop through all FASTA files in the query folder
for %%f in ("%query_folder%\*.fasta") do (
    REM Get the filename without the extension
    set "filename=%%~nf"
    
    REM Construct the output PSSM file path
    set "output_pssm=%output_folder%\!filename!.pssm"
    
    REM Check if the PSSM file already exists
    if not exist "!output_pssm!" (
        REM Run psiblast command
        psiblast -query "%%f" -db "%db_path%" -evalue %evalue% -num_iterations %iterations% -out_ascii_pssm "!output_pssm!"
    ) else (
        echo PSSM file for !filename! already exists, skipping.
    )
)

echo All PSSM files generated successfully.
pause
