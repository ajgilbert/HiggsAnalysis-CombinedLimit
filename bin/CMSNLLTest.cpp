#include <Fit/Fitter.h>
#include <TString.h>
#include <iostream>
#include <Math/MinimizerOptions.h>
#include <Minuit2/MnPrint.h>
#include <Math/IOptions.h>
#include "TRandom3.h"
#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"
#include "../interface/CMSNLLTest.h"

void TestAnalyticGrad() {
  auto func = ROOT::Math::Functor1D([](double x) { return x * x; });

  auto func_dx = ROOT::Math::Functor1D([](double x) { return 2 * x; });

  ROOT::Math::RichardsonDerivator d;
  for (double x : {1.E0, 1.E-5, 1.E+5, 1.E+9, -555.}) {
    CheckGrad(d, func, func_dx, x);
  }
}

int main(int argc, char* argv[]) {
  unsigned gen_pars = 10;
  TRandom3 rng;
  CMSNLL nllf;

  for (unsigned i = 0; i < gen_pars; ++i) {
    nllf.AddGaussianConstraint(rng.Gaus(0, 1), 1);
  }
  ROOT::Fit::Fitter fitter;
  fitter.Config().SetParamsSettings(nllf.GetParameters());
  // fitter.SetFCN(nllf);
  fitter.SetFCN((ROOT::Math::IMultiGenFunction&)nllf);

  fitter.Config().SetMinimizer("Minuit2", "Migrad");
  auto & opts = fitter.Config().MinimizerOptions();
  auto & extra_opts = ROOT::Math::MinimizerOptions::Default("Minuit2");
  extra_opts.SetValue("StorageLevel", 0);
  opts.SetErrorDef(0.5);
  opts.SetPrintLevel(10);
  opts.SetStrategy(0);
  fitter.FitFCN();
  fitter.CalculateHessErrors();

  return 0;
}
