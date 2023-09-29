#include "../interface/CombineHarvester.h"
#include "../interface/Process.h"
#include "../interface/Utilities.h"
#include "TH1F.h"
#include "TRandom3.h"
#include <Fit/Fitter.h>
#include <Math/IOptions.h>
#include <Math/MinimizerOptions.h>
#include <Minuit2/MnPrint.h>
#include <TStopwatch.h>
#include <TString.h>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <math.h>
#include <vector>

#include "../interface/CMSNLLTest.h"
#include "../interface/json.hpp"
#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"

#include "Math/Factory.h"
#include "Minuit2/Minuit2Minimizer.h"
#include <Eigen/Dense>

using json = nlohmann::json;
using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::ArrayXXd;
using Eigen::ArrayXXi;
using std::vector;

struct ProcNorms {
  ProcNorms(int Nx, int Np, int Nl, int Na) : Nx_(Nx), Np_(Np), Nl_(Nl), Na_(Na) {
    x_ = ArrayXd::Zero(Nx_);
    N0_ = ArrayXd::Zero(Np_);
    N_ = ArrayXd::Zero(Np_);
    lKappa_ = ArrayXXd::Zero(Np_, Nl_);
    Xl_ = ArrayXi::Constant(Nl_, -1);
    Lx_ = ArrayXi::Constant(Nx_, -1);
    logVal_ = ArrayXd::Zero(Np_);
    lNorm_ = ArrayXd::Zero(Np_);
    aKappaHi_ = ArrayXXd::Zero(Np_, Na_);
    aKappaLo_ = ArrayXXd::Zero(Np_, Na_);
    aAvg_ = ArrayXXd::Zero(Np_, Na_);
    aHalfDiff_ = ArrayXXd::Zero(Np_, Na_);
    aKappa_ = ArrayXd::Zero(Np_);
    dAKappa_ = ArrayXd::Zero(Np_);
    d2AKappa_ = ArrayXd::Zero(Np_);
    Xa_ = ArrayXi::Constant(Na_, -1);
    Ax_ = ArrayXi::Constant(Nx_, -1);
    dNi_ = ArrayXXd::Zero(Np_, Nx_);
    dlogVali_ = ArrayXXd::Zero(Np_, Nx_);
    dNij_ = ArrayXd::Zero(Np_);
  }
  int Nx_; // Number of parameters
  int Np_; // Number of processes
  int Nl_; // Number of parameters with a symmetric log-normal effect
  int Na_; // Number of parameters with an asymmetric log-normal effect

  ArrayXd x_; // [Nx] Parameter values

  ArrayXd N0_;      // [Np] Nominal process yields
  ArrayXd N_;       // [Np] Evaluated process yields

  ArrayXXd lKappa_; // [Np][Nl] symm. logKappa values
  ArrayXi Xl_;      // [Nl] x index of symm. lnN parameter l
  ArrayXi Lx_;      // [Nx] l index of parameter x (-1 if not in l)
  ArrayXd logVal_;  // [Np] sum(x*logkappa)
  ArrayXd lNorm_;   // [Np] exp(sum(x*lnk))

  // Fixed data
  ArrayXXd aKappaHi_; // [Np][Na] asymm. logKappaHi values
  ArrayXXd aKappaLo_; // [Np][Na] asymm. logKappaLo values
  ArrayXXd aAvg_; // [Np][Na]
  ArrayXXd aHalfDiff_; // [Np][Na]

  // Scratch space
  ArrayXd aKappa_; // [Np]
  ArrayXd dAKappa_; // [Np]
  ArrayXd d2AKappa_; // [Np]

  // Fixed data
  ArrayXi Xa_;      // [Na] x index of asymm. lnN parameter a
  ArrayXi Ax_;      // [Nx] a index of parameter x (-1 if not in l)

  ArrayXXd dNi_; // [Np][Nx] First derivative of N wrt. parameters i
  ArrayXXd dlogVali_; // [Np][Nx] First derivative of logVal wrt. parameters i
  ArrayXd dNij_; // [Np] Second derivative of n wrt. paraemters i and j

  ArrayXd & aKappa(double x, unsigned ia) {
    logKappaMorph(x, aKappa_, aKappaHi_.col(ia), aKappaLo_.col(ia), aAvg_.col(ia), aHalfDiff_.col(ia));
    return aKappa_;
  }

  ArrayXd & dAKappa(double x, unsigned ia) {
    gradLogKappaMorph(x, dAKappa_, aHalfDiff_.col(ia));
    return dAKappa_;
  }

  ArrayXd & d2AKappa(double x, unsigned ia) {
    g2LogKappaMorph(x, d2AKappa_, aHalfDiff_.col(ia));
    return d2AKappa_;
  }

  /*
    Formula for mixed partial derivative of F = G * H
    dFij = dGij*H + dGi*dHj + dGj*dHi + G*dHij
    => Asumme H:=N, my caller will want go calculate some Fij
    => I need to give him: N, dNi, dNj, dNij

    My caller is probably calculating the full Hessian, which
    means he's going to ask me for the same dNi, dNj over and over
    again.
    => The dNi should probably be cached, but by whom?
    => Implies an array of [Np][Nx]

  */


  void SetCaches() {
    aAvg_ = 0.5 * (aKappaHi_ - aKappaLo_);
    aHalfDiff_ = 0.5 * (aKappaHi_ + aKappaLo_);
  }

  // Note a trick using Eigen::Ref here for the logKappa (which will be modified)
  // Eigen doesn't allow blocks of ArrayXXd to be passed through to ArrayXd as a 
  // non-const ref, but this does the job
  void logKappaMorph(double x, Eigen::Ref<ArrayXd> logKappa, ArrayXd const& kHi, ArrayXd const& kLo, ArrayXd const& avg, ArrayXd const& halfdiff) {
    if (fabs(x) >= 0.5) {
      if (x >= 0) {
        logKappa = kHi;
      } else {
        logKappa = -kLo;
      }
    } else {
      double twox = x + x;
      double twox2 = twox * twox;
      double alpha = 0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.);   
      logKappa = avg + alpha * halfdiff; 
    }
  }

  void gradLogKappaMorph(double x, Eigen::Ref<ArrayXd> dLogKappa, ArrayXd const& halfdiff) {
    if (fabs(x) >= 0.5) {
      dLogKappa = 0.;
    } else {
      double twox = x + x;
      double twox2 = twox * twox;
      double alpha = 0.125 * (twox2 * (twox2 * 30. - 60.)  + 30.);
      dLogKappa = alpha * halfdiff; 
    }
  }

  void g2LogKappaMorph(double x, Eigen::Ref<ArrayXd> dLogKappa, ArrayXd const& halfdiff) {
    if (fabs(x) >= 0.5) {
      dLogKappa = 0.;
    } else {
      double twox = x + x;
      double twox2 = twox * twox;
      double alpha = 30. * twox * (twox2 - 1.);
      dLogKappa = alpha * halfdiff; 
    }
  }

  void Eval() {
    SetCaches();
    logVal_ = 0.;
    for (int il = 0; il < Nl_; ++il) {
      logVal_ += x_[Xl_[il]] * lKappa_.col(il);
    }

    for (int ia = 0; ia < Na_; ++ia) {
      double x = x_[Xa_[ia]];
      logVal_ += x * aKappa(x, ia);
    }

    lNorm_ = Eigen::exp(logVal_);
    N_ = N0_ * lNorm_;
  }

  void Gradients() {
    // ! Assumes Eval() has been called
    dNi_ = 0.;

    dlogVali_ = 0.;

    // Gradient is N * d(logVal_)
    // Let's start by calcuating d(logVal_) in dNi_:
    for (int il = 0; il < Nl_; ++il) {
      int iX = Xl_[il];
      dlogVali_.col(iX) += lKappa_.col(il);
    }
    for (int ia = 0; ia < Na_; ++ia) {
      int iX = Xa_[ia];
      double x = x_[iX];
      dlogVali_.col(iX) += (aKappa(x, ia) + x * dAKappa(x, ia));
    }

    // Finish up by multiplying by N_
    for (int ix = 0; ix < Nx_; ++ix) {
      dNi_.col(ix) = N_ * dlogVali_.col(ix);
    }
  }

  void HessianElement(int i, int j) {
    // dNij = N0 * dij(exp(logVal_)) = dj(N0 * exp(logVal_) * di(logVal_))
    //                               = N0 * [dj(exp(logVal_))*di(logVal_) + ]
    // so, 
    // dij(logVal_) = dij(x*lKappa_ + x*aKappa_)
    //              
    dNij_ = dlogVali_.col(i) * dlogVali_.col(j);
    if ((i == j) && Ax_[i] >= 0) {
      double x = x_[i];
      dNij_ += (2. * dAKappa(x, Ax_[i]) + x * d2AKappa(x, Ax_[i]));
    }
    dNij_ *= N_;
  }
};

std::vector<double> BinVec(TH1 &h) {
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

void QuickPrint(ProcNorms & pn) {
  pn.Eval();
  pn.Gradients();
  std::cout << "x = " << pn.x_.transpose() << ", N = " << pn.N_.transpose()
            << std::endl;
  std::cout << "lKappa = \n" << pn.lKappa_ << std::endl;
  std::cout << "aKappaLo = \n" << pn.aKappaLo_ << std::endl;
  std::cout << "aKappaHi = \n" << pn.aKappaHi_ << std::endl;
  std::cout << "aKappa = \n" << pn.aKappa_ << std::endl;
  std::cout << "dNi = \n" << pn.dNi_ << std::endl;
  for (int i = 0; i < pn.Nx_; ++i) {
    for (int j = 0; j <= i; ++j) {
      pn.HessianElement(i, j);
      std::cout << "Hessian(" << i << ", " << j
                << ") = " << pn.dNij_.transpose() << std::endl;
    }
  }
}


void SymTest() {
  ProcNorms pn(2, 3, 2, 0);
  pn.N0_ << 10., 20., 30.;
  pn.lKappa_ << std::log(1.1), std::log(1.3),
                std::log(1.1), 0.,
                0.,               std::log(1.5);
  pn.Lx_ << 0, 1;
  pn.Xl_ << 0, 1;

  QuickPrint(pn);

  pn.x_ << 1., 1.;
  QuickPrint(pn);

  pn.x_ << 1., -1.;
  QuickPrint(pn);
}

ArrayXd NumericGrad(ProcNorms & pn, unsigned ix) {
  ROOT::Math::RichardsonDerivator d;
  double test_x = pn.x_[ix];
  ArrayXd res = ArrayXd::Zero(pn.N_.size());
  for (int iN = 0; iN < pn.N_.size(); ++iN) {
    auto func = ROOT::Math::Functor1D([&](double x) { 
      pn.x_[ix] = x;
      pn.Eval();
      return pn.N_[iN];
    });
    res[iN] = d.Derivative1(func, test_x, 0.001);
  }
  return res;
}

void AsymTest(double test0, double test1) {
  ProcNorms pn(2, 2, 1, 2);
  pn.N0_ << 10., 20.;
  pn.aKappaLo_ << std::log(0.7), std::log(0.9),
                  std::log(0.7), 0.;
  pn.aKappaHi_ << std::log(1.1), std::log(1.2),
                  std::log(1.1), 0.;
  pn.lKappa_ << 0.,
                std::log(1.03);
  pn.Ax_ << 0, 1;
  pn.Xa_ << 0, 1;
  pn.Lx_ << -1, 0;
  pn.Xl_ << 1;

  pn.x_ << test0, test1;

  QuickPrint(pn);

  for (int ix = 0; ix < pn.x_.size(); ++ix) {
    std::cout << "Numeric dN[" << ix << "]:\n" << NumericGrad(pn, ix) << std::endl;
    pn.x_ << test0, test1;
  }

  pn.x_ << test0, test1;

  QuickPrint(pn);

  for (int ix = 0; ix < pn.x_.size(); ++ix) {
    std::cout << "Numeric dN[" << ix << "]:\n" << NumericGrad(pn, ix) << std::endl;
    pn.x_ << test0, test1;
  }

  // ROOT::Math::RichardsonDerivator d;

  // auto func = ROOT::Math::Functor1D([&](double x) { 
  //   pn.x_ << x;
  //   pn.Eval();
  //   return pn.N_[0];
  // });

  // auto res = d.Derivative1(func, test_x, 0.001);
  // std::cout << "Numeric dN[0] = " << res << std::endl;

  // auto dfunc = ROOT::Math::Functor1D([&](double x) { 
  //   return d.Derivative1(func, x, 0.001);
  // });

  // ROOT::Math::RichardsonDerivator d2;

  // auto res2 = d2.Derivative1(dfunc, test_x, 0.001);
  // std::cout << "Numeric d2N[0] = " << res2 << std::endl;


  // pn.x_ << 1.;
  // QuickPrint(pn);

  // pn.x_ << -1.;
  // QuickPrint(pn);
}

int main(int argc, char *argv[]) {
  AsymTest(0, 0.);
  return 0;
}
