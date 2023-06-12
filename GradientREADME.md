# Testing analytic gradients

This branch contains an (in development) test of analytic gradients for the standard binned likelihood model used in combine.

At the moment, the model is reimplemented from scratch, currently independently from RooFit and combine, and interfaces directly with the ROOT minimisation interface. The eventual plan is to wrap this code in a way that is accessible to RooFit.

Due to some important updates/fixes for the Minuit2 Migrad algorithm when using analytic gradients, it is best to use this code against ROOT v6.28, which is not currently available in a CMSSW release. Instead, we use a recent LCG release:

```sh
. env_lcg.sh
make -j8
```

A test program `CMSNLLTest` can be used to run a fit on a datacard:

```
CMSNLLTest '{"bigmodel": false, "grad": false, "card":"comb_2021_tth_multilepton_cleaned.txt", "strategy":0, "tol":0.1, "printLevel": 2}'
```

The options are specified in JSON format:

 - `"bigmodel"`: set this to `false` for now
 - `"grad"`: whether to use analytic gradient for now
 - `"card"`: input datacard, see supported features below
 - `"strategy"`: usual Minuit strategy setting
 - `"tol"`: usual Minuit fit EDM tolerance setting

Datacard/model features currently planned/supported:
 - [x] TH1 input templates
 - [x] Symmetric lnN uncertainties
 - [ ] Asymmetric lnN uncertainties
 - [ ] `shape[N]` uncertainties
 - [ ] Single-parameter rateParams
 - [ ] RooFormula rateParams
 - [ ] autoMCStats parameters

 Make sure the input datacards only include the features that are supported. Some example cards are available at `/afs/cern.ch/work/a/agilbert/public/nll-tests/`.



