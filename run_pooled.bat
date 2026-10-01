@echo off
REM ---------------------------------------------------------------------------
REM PREREGISTER Amendment 6, Experiment A -- does training-set SIZE explain the
REM cross-database transfer gap?
REM
REM Three arms on ONE shared label space (SR, SB, ST, AFIB/AFL), so the only
REM variable that moves is how many training records there are:
REM
REM     chapman4    4,697 records
REM     ningbo4    15,164 records
REM     pooled     19,861 records
REM
REM     conda activate <your-env>
REM     cd /d <path-to-this-repo>
REM     run_pooled.bat
REM
REM Resumable: every step is skipped if its output already exists, so a crash or
REM a Ctrl-C costs only the step it was on. Expect roughly 8-10 hours in total,
REM most of it the three teacher grids. Run it overnight.
REM ---------------------------------------------------------------------------

setlocal enabledelayedexpansion

echo Start: %DATE% %TIME%
echo.

for %%D in (chapman4 ningbo4 pooled) do (

    echo ==========================================================
    echo   ARM %%D
    echo ==========================================================

    REM ---- teachers, one per fold ------------------------------------------
    for %%F in (0 1 2 3 4) do (
        if exist "checkpoints\teacherv2_%%D_without_others_fold%%F.pt" (
            echo   teacher fold %%F  -- already trained, skipped
        ) else (
            echo   teacher fold %%F  -- training
            python teacher_v2.py --dataset %%D --fold %%F --epochs 60 --batch 32
            if errorlevel 1 ( echo FAILED: teacher %%D fold %%F & exit /b 1 )
        )
    )

    REM ---- freeze each teacher to cached logits and embeddings --------------
    for %%F in (0 1 2 3 4) do (
        if exist "cache\teacherv2_%%D_without_others_fold%%F.npz" (
            echo   export  fold %%F  -- already cached, skipped
        ) else (
            echo   export  fold %%F
            python export_teacher_v2.py --dataset %%D --fold %%F
            if errorlevel 1 ( echo FAILED: export %%D fold %%F & exit /b 1 )
        )
    )

    REM ---- student grid: 2 modes x 5 folds x 5 seeds ------------------------
    REM RKD is not run. It was refuted on Chapman (-0.47, p=0.0073) and on CPSC
    REM (-3.48, p<0.0001); repeating it here would only spend GPU time to
    REM confirm a closed question.
    for %%F in (0 1 2 3 4) do (
        for %%S in (0 1 2 3 4) do (
            if exist "results\student_ce_%%D_without_others_fold%%F_seed%%S.json" (
                echo   ce  fold %%F seed %%S  -- skipped
            ) else (
                echo   ce  fold %%F seed %%S
                python train_student.py --mode ce --dataset %%D --fold %%F --seed %%S >nul 2>&1
                if errorlevel 1 ( echo FAILED: ce %%D fold %%F seed %%S & exit /b 1 )
            )
            if exist "results\student_kd_v2_%%D_without_others_fold%%F_seed%%S.json" (
                echo   kd  fold %%F seed %%S  -- skipped
            ) else (
                echo   kd  fold %%F seed %%S
                python train_student.py --mode kd --dataset %%D --fold %%F --seed %%S --teacher v2 >nul 2>&1
                if errorlevel 1 ( echo FAILED: kd %%D fold %%F seed %%S & exit /b 1 )
            )
        )
    )

    echo.
)

echo Training done: %DATE% %TIME%
echo.

REM ---- in-domain analysis, one arm at a time --------------------------------
for %%D in (chapman4 ningbo4 pooled) do (
    echo ===== IN DOMAIN: %%D =====
    python analyze.py --dataset %%D --teacher v2
    echo.
)

REM ---- the actual test: external transfer, three arms, same targets ---------
REM Primary endpoint is AFIB/AFL F1 on CPSC, pooled versus ningbo4.
for %%D in (chapman4 ningbo4 pooled) do (
    for %%T in (cpsc cpsc_nofilter georgia) do (
        echo ===== EXTERNAL: %%D -^> %%T =====
        python evaluate_external.py --source %%D --target %%T --teacher v2
        echo.
    )
)

echo Finished: %DATE% %TIME%

endlocal
