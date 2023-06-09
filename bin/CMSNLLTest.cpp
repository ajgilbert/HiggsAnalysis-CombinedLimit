#include <Fit/Fitter.h>
#include <TString.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <Math/MinimizerOptions.h>
#include <Minuit2/MnPrint.h>
#include <Math/IOptions.h>
#include "../interface/CombineHarvester.h"
#include "../interface/Process.h"
#include "../interface/Utilities.h"
#include "TRandom3.h"
#include <TStopwatch.h>
#include "TH1F.h"

#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"
#include "../interface/CMSNLLTest.h"
#include "../interface/json.hpp"

#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Factory.h"

using json = nlohmann::json;

std::vector<double> BinVec(TH1 & h) {
  std::vector<double> res(h.GetNbinsX());
  for (int i = 1; i < h.GetNbinsX() + 1; ++i) {
    res[i - 1] = h.GetBinContent(i);
  }
  return res;
}

void TestAnalyticGrad() {
  auto func = ROOT::Math::Functor1D([](double x) { return x * x; });

  auto func_dx = ROOT::Math::Functor1D([](double x) { return 2 * x; });

  ROOT::Math::RichardsonDerivator d;
  for (double x : {1.E0, 1.E-5, 1.E+5, 1.E+9, -555.}) {
    CheckGrad(d, func, func_dx, x);
  }
}

int main(int argc, char* argv[]) {
  json js = json::parse(argv[1]);
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

  ch::CombineHarvester cmb;

  bool bigmodel = js.at("bigmodel");
  std::string readcard = "";
  if (js.count("card")) {
    readcard = js.at("card");
    cmb.ParseDatacard(readcard, "");
    cmb.PrintAll();
    nllf.AddParameter("r", 1., 0.1, 0., 10.);

    std::set<std::string> added_systs;
    for (auto const& b : cmb.bin_set()) {
      TH1F h = cmb.cp().bin({b}).GetObservedShape();
      std::cout << h.GetNbinsX() << "\n";
      h.Print("range");
      if (h.GetNbinsX() == 1) {
        // This is a counting card
        h = TH1F("obs", "", 1, 0, 1);
        h.SetBinContent(1, cmb.cp().bin({b}).GetObservedRate());
      }
      h.Print("range");
      std::vector<ch::Process *> procs;
      cmb.cp().bin({b}).ForEachProc([&](ch::Process *p) {
        procs.push_back(p);
      });

      std::map<std::string, std::pair<std::vector<unsigned >, std::vector<double>>> syst_vals;
      std::vector<unsigned> sig_procs;
      // auto procs = ch::Set2Vec(cmb.cp().bin({b}).process_set());
      unsigned ic = nllf.AddChannel(h.GetNbinsX(), procs.size());
      nllf.SetData(ic, BinVec(h));
      for (unsigned ip = 0; ip < procs.size(); ++ip) {
        // std::vector<double> vals;
        auto phist = procs[ip]->ClonedScaledShape();
        if (phist.get() == nullptr) {
          phist = std::make_unique<TH1F>("proc", "", 1, 0, 1);
          phist->SetBinContent(1, procs[ip]->no_norm_rate());
        }
        nllf.SetTemplate(ic, ip, BinVec(*phist));
        if (procs[ip]->signal()) {
          sig_procs.push_back(ip);
        }
        cmb.cp().bin({b}).process({procs[ip]->process()}).ForEachSyst([&](ch::Systematic *s) {
          if (s->type() == "lnN" && s->asymm() == false) {
            std::cout << s->name() << "\t" << s->value_u() << "\n";
            if (added_systs.count(s->name()) == 0) {
              nllf.AddParameter(s->name(), 0., 0.5, -7., 7.);
              nllf.AddGaussianConstraint(nllf.par(s->name()), 0., 1.);
              added_systs.insert(s->name());
            }
            syst_vals[s->name()].first.push_back(ip);
            syst_vals[s->name()].second.push_back(s->value_u());
          }
        });
      }
      nllf.AddRateParam(nllf.par("r"), ic, sig_procs);
      for (auto const& s : syst_vals) {
        nllf.AddLogNormal(nllf.par(s.first), ic, s.second.first, s.second.second);
      }
    }
    nllf.evaluate(true);
    nllf.PrintModel();
    // std::exit(0);
  } else if (bigmodel) {
    unsigned n_bins = 2000;
    unsigned n_sig = 50;
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
      nllf.AddParameter(name, 1., 1., -10., 10.);
      nllf.AddRateParam(nllf.par(name), 0, {is + 1});
    }
    nllf.SetData(ic, data);
    nllf.evaluate(true);

  } else {
    /*
    unsigned ic = nllf.AddChannel(1, 2);
    nllf.SetTemplate(ic, 0, {10.});
    nllf.SetTemplate(ic, 1, {1.});
    nllf.SetData(ic, {12.});
    nllf.AddParameter("r", 1.);
    nllf.AddParameter("scale", 1.);
    nllf.AddGaussianConstraint(nllf.par("scale"), 2., 1.);
    nllf.AddRateParam(nllf.par("r"), 0, {1});
    nllf.evaluate(true);
    nllf.PrintModel();
    */

    unsigned ic = nllf.AddChannel(1, 2);
    nllf.SetTemplate(ic, 0, {45.});
    nllf.SetTemplate(ic, 1, {55.});
    nllf.SetData(ic, {80.});
    nllf.AddParameter("scale", 0., 1., -7., 7.);
    nllf.AddGaussianConstraint(nllf.par("scale"), 0., 1.);
    nllf.AddLogNormal(nllf.par("scale"), 0, {0}, {1.5});

    nllf.AddParameter("scale2", 0., 1., -7., 7.);
    nllf.AddGaussianConstraint(nllf.par("scale2"), 0., 1.);
    nllf.AddLogNormal(nllf.par("scale2"), 0, {0, 1}, {1.1, 1.4});


    // nllf.AddRateParam(nllf.par("r"), 0, {1});
    nllf.evaluate(true);
    nllf.PrintModel();
    exit(0);

  }


  // nllf.AddParameter("r2", 0.5);
  // nllf.AddRateParam(nllf.par("r2"), 0, {0, 1});
  // nllf.AddRateParam(nllf.par("r2"), 1, {0});
  // ic = nllf.AddChannel(5, 2);
  // nllf.SetTemplate(ic, 0, {1., 2., 3., 4., 5.});
  // nllf.SetTemplate(ic, 1, {0., 0.5, 0.7, 0.5, 0.});
  // nllf.SetData(ic, {2., 3., 4., 4., 5.});
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
  bool use_grad = js.at("grad");
  if (use_grad) {
    fitter.SetFCN(nllf);
  } else {
    fitter.SetFCN((ROOT::Math::IMultiGenFunction&)nllf);
  }

  fitter.Config().SetMinimizer("Minuit2", "Migrad");
  auto& opts = fitter.Config().MinimizerOptions();
  auto& extra_opts = ROOT::Math::MinimizerOptions::Default("Minuit2");
  extra_opts.SetValue("StorageLevel", 0);
  opts.SetErrorDef(0.5);
  opts.SetPrintLevel(js.at("printLevel"));
  opts.SetStrategy(js.at("strategy"));
  opts.SetTolerance(js.at("tol"));

  TStopwatch tw;
  tw.Start();
  // std::cout << fitter.GetMinimizer() << "\n";
  // fitter.CalculateHessErrors();
  fitter.FitFCN();
  // fitter.CalculateHessErrors();
  // fitter.FitFCN();
  // ROOT::Minuit2::Minuit2Minimizer *minim = dynamic_cast<ROOT::Minuit2::Minuit2Minimizer*>(fitter.GetMinimizer());
  // minim->Minimize();
  // fitter.SetFCN((ROOT::Math::IMultiGenFunction&)nllf);
  
  // minim->
  // minim->Hesse();

  // fitter.FitFCN();
  // fitter.CalculateMinosErrors();
  tw.Stop();
  fitter.Result().Print(std::cout);
  std::cout << ">> Fit in " << tw.RealTime() << std::endl;
  // nllf.PrintModel();
  // fitter.CalculateHessErrors();

  return 0;
}
