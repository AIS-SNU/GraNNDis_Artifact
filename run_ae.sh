echo "Preparing Log Directory"
rm -rf Logs
mkdir Logs

echo "Run AE1"
cd AE
python ae1_tpt_opt_baseline.py
python ae1_tpt_flx.py
python ae1_tpt_cob.py
python ae1_tpt_eas.py

echo "Run AE2"
python ae2_acc_baseline.py
python ae2_acc_eas.py

cd ..