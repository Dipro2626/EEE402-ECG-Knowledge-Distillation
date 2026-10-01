@echo off
REM ---------------------------------------------------------------------------
REM Full pipeline: Teacher v2 -> frozen cache -> student grid -> analysis ->
REM external evaluation -> tables.
REM
REM     conda activate <your-env>
REM     cd /d <path-to-this-repo>
REM     run_all.bat
REM
REM Resumable: each step is skipped when its output already exists.
REM The pooled / four-class experiment is separate: run_pooled.bat.
REM ---------------------------------------------------------------------------

setlocal enabledelayedexpansion
echo Start: %DATE% %TIME%

REM dataset:min_class  (Ningbo drops SVT/AT, 15 records, via --min_class 100)
for %%A in (chapman:0 ningbo:100) do (
    for /f "tokens=1,2 delims=:" %%D in ("%%A") do (
        set DS=%%D
        set MC=%%E
        set TAG=%%D_without_others
        if not "%%E"=="0" set TAG=%%D_without_others_min%%E

        echo ==================== !DS! ====================

        for %%F in (0 1 2 3 4) do (
            if not exist "checkpoints\teacherv2_!TAG!_fold%%F.pt" (
                python teacher_v2.py --dataset !DS! --min_class !MC! --fold %%F
                if errorlevel 1 ( echo FAILED teacher !DS! fold %%F & exit /b 1 )
            )
            if not exist "cache\teacherv2_!TAG!_fold%%F.npz" (
                python export_teacher_v2.py --dataset !DS! --min_class !MC! --fold %%F
                if errorlevel 1 ( echo FAILED export !DS! fold %%F & exit /b 1 )
            )
        )

        for %%F in (0 1 2 3 4) do (
            for %%S in (0 1 2 3 4) do (
                if not exist "results\student_ce_!DS!_without_others_fold%%F_seed%%S.json" (
                    python train_student.py --mode ce --dataset !DS! --min_class !MC! --fold %%F --seed %%S >nul 2>&1
                    if errorlevel 1 ( echo FAILED ce !DS! %%F %%S & exit /b 1 )
                )
                if not exist "results\student_kd_v2_!DS!_without_others_fold%%F_seed%%S.json" (
                    python train_student.py --mode kd --dataset !DS! --min_class !MC! --fold %%F --seed %%S >nul 2>&1
                    if errorlevel 1 ( echo FAILED kd !DS! %%F %%S & exit /b 1 )
                )
            )
        )
        python analyze.py --dataset !DS! --min_class !MC!
    )
)

echo ==================== external ====================
for %%T in (ningbo cpsc cpsc_nofilter georgia) do (
    python evaluate_external.py --source chapman --target %%T
)
for %%T in (chapman cpsc cpsc_nofilter georgia) do (
    python evaluate_external.py --source ningbo --target %%T --min_class 100
)
python evaluate_external.py --source ningbo --target mitbih --min_class 100 --save_preds
python analyze_mitbih.py

echo ==================== deployment ====================
python benchmark.py

echo ==================== tables and figures ====================
python make_all_tables.py
python scripts\summarize_results.py
python scripts\make_figures.py

echo Finished: %DATE% %TIME%
endlocal
