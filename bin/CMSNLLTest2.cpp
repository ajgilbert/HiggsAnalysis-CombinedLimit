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
  ProcNorms(int Nx, int Np, int Nl) : Nx_(Nx), Np_(Np), Nl_(Nl) {
    x_ = ArrayXd::Zero(Nx_);
    N0_ = ArrayXd::Zero(Np_);
    N_ = ArrayXd::Zero(Np_);
    lNorm_ = ArrayXd::Zero(Np_);
    logVal_ = ArrayXd::Zero(Np_);
    lKappa_ = ArrayXXd::Zero(Np_, Nl_);
    Xl_ = ArrayXi::Constant(Nl_, -1);
    Lx_ = ArrayXi::Constant(Nx_, -1);
    dNi_ = ArrayXXd::Zero(Np_, Nx_);
    dNij_ = ArrayXd::Zero(Np_);
  }
  int Nx_; // Number of parameters
  int Np_; // Number of processes
  int Nl_; // Number of parameters with a symmetric log-normal effect

  ArrayXd x_; // [Nx] Parameter values

  ArrayXd N0_;      // [Np] Nominal process yields
  ArrayXd N_;       // [Np] Evaluated process yields
  ArrayXd lNorm_;   // [Np] exp(sum(x*lnk))
  ArrayXd logVal_;  // [Np] sum(x*logkappa)
  ArrayXXd lKappa_; // [Np][Nl] symm. logKappa values
  ArrayXi Xl_;      // [Nl] x index of symm. lnN parameter l
  ArrayXi Lx_;      // [Nx] l index of parameter x (-1 if not in l)

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
  ArrayXXd dNi_; // [Np][Nx] First derivative of N wrt. parameters i
  ArrayXd dNij_; // [Np] Second derivative of n wrt. paraemters i and j

  void Eval() {
    logVal_ = 0.;
    for (int il = 0; il < Nl_; ++il) {
      logVal_ += x_[Xl_[il]] * lKappa_.col(il);
    }
    lNorm_ = Eigen::exp(logVal_);
    N_ = N0_ * lNorm_;
  }

  void Gradients() {
    // ! Assumes Eval() has been called
    dNi_ = 0.;
    for (int il = 0; il < Nl_; ++il) {
      int iX = Xl_[il];
      dNi_.col(iX) = N_ * lKappa_.col(il);
    }
  }
  void HessianElement(int i, int j) {
    dNij_ = N_;
    MultLogKappaForX(i);
    MultLogKappaForX(j);
  }

  void MultLogKappaForX(int k) {
    if (Lx_[k] >= 0) {
      dNij_ *= lKappa_.col(Lx_[k]);
    } else {
      dNij_ *= 0.;
    }
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

int main(int argc, char *argv[]) {
  ProcNorms pn(2, 3, 2);
  pn.N0_ << 10., 20., 30.;
  pn.lKappa_ << std::log(1.1), std::log(1.3), std::log(1.1), 0., 0.,
      std::log(1.5);
  pn.Lx_ << 0, 1;
  pn.Xl_ << 0, 1;

  auto QuickPrint = [](ProcNorms &pn) {
    pn.Eval();
    pn.Gradients();
    std::cout << "x = " << pn.x_.transpose() << ", N = " << pn.N_.transpose()
              << std::endl;
    std::cout << "dNi = \n" << pn.dNi_ << std::endl;
    for (int i = 0; i < pn.Nx_; ++i) {
      for (int j = 0; j <= i; ++j) {
        pn.HessianElement(i, j);
        std::cout << "Hessian(" << i << ", " << j
                  << ") = " << pn.dNij_.transpose() << std::endl;
      }
    }
  };

  QuickPrint(pn);

  pn.x_ << 1., 1.;
  QuickPrint(pn);

  pn.x_ << 1., -1.;
  QuickPrint(pn);

  return 0;
}
