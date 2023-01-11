#include <Fit/Fitter.h>
#include <TString.h>
#include <iostream>
#include <Math/MinimizerOptions.h>
#include <Minuit2/MnPrint.h>
#include <Math/IOptions.h>
#include "TRandom3.h"
#include <TStopwatch.h>

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
  // unsigned gen_pars_gaus = 2000;
  // unsigned gen_pars_pois = 2000;
  // TRandom3 rng;
  CMSNLL nllf;

  // for (unsigned i = 0; i < gen_pars_gaus; ++i) {
  //   nllf.AddGaussianConstraint(rng.Gaus(0, 1), 1);
  // }
  // for (unsigned i = 0; i < gen_pars_pois; ++i) {
  //   nllf.AddPoissonConstraint(std::max(1., double(int(rng.Gaus(30, 5) + 0.5))));
  // }

  unsigned ic = nllf.AddChannel(3, 2);
  nllf.SetTemplate(ic, 0, {1., 2., 3.});
  nllf.SetTemplate(ic, 1, {4., 5., 6.});
  nllf.SetData(ic, {8., 5., 8.});

  ic = nllf.AddChannel(1, 1);
  nllf.SetTemplate(ic, 0, {10.});
  nllf.SetData(ic, {9.});

  nllf.AddParameter("r", 1.);
  nllf.AddParameter("r2", 0.5);
  nllf.AddRateParam(nllf.par("r"), 0, {0});
  nllf.AddRateParam(nllf.par("r2"), 0, {0, 1});
  nllf.AddRateParam(nllf.par("r2"), 1, {0});
  // ic = nllf.AddChannel(5, 2);
  // nllf.SetTemplate(ic, 0, {1., 2., 3., 4., 5.});
  // nllf.SetTemplate(ic, 1, {0., 0.5, 0.7, 0.5, 0.});
  // nllf.SetData(ic, {2., 3., 4., 4., 5.});
  nllf.evaluate();
  nllf.PrintModel();
  // std::cout << "nll = " << nllf.evaluate() << std::endl;
  // nllf.SetTemplate(ic, 0, {10.});

  // nllf.AddLogNormal("CMS_lumi", 0, 0, 1.1);
  // nllf.AddLogNormal("CMS_jes", 0, 0, 1.25);
  // nllf.evaluate();
  // nllf.SetParameter(0, -1.);
  // nllf.SetParameter(1, 1.5);
  // nllf.evaluate();
  
  /*

  ROOT::Fit::Fitter fitter;
  fitter.Config().SetParamsSettings(nllf.GetParameters());
  nllf.SetZeroPoint(fitter.Config().ParamsValues().data());
  fitter.SetFCN(nllf);
  // fitter.SetFCN((ROOT::Math::IMultiGenFunction&)nllf);

  fitter.Config().SetMinimizer("Minuit2", "Migrad");
  auto & opts = fitter.Config().MinimizerOptions();
  auto & extra_opts = ROOT::Math::MinimizerOptions::Default("Minuit2");
  extra_opts.SetValue("StorageLevel", 0);
  opts.SetErrorDef(0.5);
  opts.SetPrintLevel(10);
  opts.SetStrategy(0);
  TStopwatch tw;
  tw.Start();
  fitter.FitFCN();
  tw.Stop();
  std::cout << ">> Fit in " << tw.RealTime() << std::endl;
  // fitter.CalculateHessErrors();
  */

  return 0;
}
