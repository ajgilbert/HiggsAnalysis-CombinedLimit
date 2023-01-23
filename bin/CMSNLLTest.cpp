#include <Fit/Fitter.h>
#include <TString.h>
#include <iostream>
#include <Math/MinimizerOptions.h>
#include <Minuit2/MnPrint.h>
#include <Math/IOptions.h>
#include "TRandom3.h"
#include <TStopwatch.h>
#include "TH1F.h"

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

  TRandom3 rng;

  unsigned n_bins = 20000;
  unsigned n_sig = 500;

  std::vector<double> bkg(n_bins);
  std::vector<std::vector<double>> sig(n_sig);
  std::vector<double> data(n_bins);
  for (unsigned is = 0; is < n_sig; ++is) {
    sig[is] = std::vector<double>(n_bins);
  }
  for (unsigned i = 0; i < n_bins; ++i) {
    double sum = 0.;
    bkg[i] = rng.Gaus(50., 4.);
    sum += bkg[i];
    for (unsigned is = 0; is < n_sig; ++is) {
      int ibegin = is * (n_bins / n_sig) - 5;
      int iend = (is + 1) * (n_bins / n_sig) + 5;
      if ((int)i >= ibegin && (int)i <= iend) {
        sig[is][i] = rng.Gaus(2., 0.2);
        sum += sig[is][i];
      }
    }
    // sig[i] = rng.Gaus(5, 0.5);
    data[i] = rng.PoissonD(sum);

  }

  unsigned ic = nllf.AddChannel(n_bins, 1 + n_sig);
  nllf.SetTemplate(ic, 0, bkg);
  for (unsigned is = 0; is < n_sig; ++is) {
    nllf.SetTemplate(ic, is + 1, sig[is]);
    std::string name(TString::Format("r%i", is).Data());
    nllf.AddParameter(name, 1.);
    nllf.AddRateParam(nllf.par(name), 0, {is + 1});
  }
  nllf.SetData(ic, data);

  // ic = nllf.AddChannel(1, 1);
  // nllf.SetTemplate(ic, 0, {10.});
  // nllf.SetData(ic, {9.});

  // nllf.AddParameter("r2", 0.5);
  // nllf.AddRateParam(nllf.par("r2"), 0, {0, 1});
  // nllf.AddRateParam(nllf.par("r2"), 1, {0});
  // ic = nllf.AddChannel(5, 2);
  // nllf.SetTemplate(ic, 0, {1., 2., 3., 4., 5.});
  // nllf.SetTemplate(ic, 1, {0., 0.5, 0.7, 0.5, 0.});
  // nllf.SetData(ic, {2., 3., 4., 4., 5.});
  nllf.evaluate(true);
  // nllf.PrintModel();
  // std::vector<double> x{10.};
  // std::vector<double> g(nllf.NDim(), 0.);
  // std::cout << "nll = " << nllf.DoEval(x.data()) << std::endl;
  // nllf.Gradient(x.data(), g.data());
  // std::cout << "dnll = " << nllf.FmtVec(g, "%6.2f") << std::endl;
  // nllf.SetTemplate(ic, 0, {10.});

  // nllf.AddLogNormal("CMS_lumi", 0, 0, 1.1);
  // nllf.AddLogNormal("CMS_jes", 0, 0, 1.25);
  // nllf.evaluate();
  // nllf.SetParameter(0, -1.);
  // nllf.SetParameter(1, 1.5);
  // nllf.evaluate();
  
  // return 0;

  ROOT::Fit::Fitter fitter;
  fitter.Config().SetParamsSettings(nllf.GetParameters());
  // nllf.SetZeroPoint(fitter.Config().ParamsValues().data());
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


  return 0;
}
