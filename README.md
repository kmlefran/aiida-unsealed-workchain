# aiida-unsealed-workchain
Files for reproducing an error in AiiDA resulting in an unsealed excepted WorkChain

1. Download `aiida-aimall` 0.7.4 via `pip`
2. Put the `minimal_workchains.py` in the Python path. I suppose the import statement of DiffuseImplicitHybridWorkchain in errortest.ipynb will need to be fixed
3. Adjust the Gaussian code instance in the inputs to one you have set up
4. Download all files, executing cells in errortest.ipynb will submit the calculation, which requires Gaussian Software
5. The workchain excepts due to an error in the results step, and the workchain is unsealed
