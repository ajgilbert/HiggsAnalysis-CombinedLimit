#include "../interface/CMSNLLTest.h"
#include <iostream>
#include "TRandom3.h"


void NearlyEqual(double x1, double x2) {
  double compare = 2 * (x1 - x2) / (x1 + x2);
  Printf("%.8f %.8f %.8f\n", x1, x2, compare);
}
void CheckGrad(ROOT::Math::RichardsonDerivator& d,
               ROOT::Math::Functor1D const& f,
               ROOT::Math::Functor1D const& f_dx,
               double x) {
  NearlyEqual(d.Derivative1(f, x, 0.001), f_dx(x));
}

ROOT::Math::IMultiGradFunction* CMSNLL::Clone() const { return new CMSNLL(*this); }

unsigned int CMSNLL::NDim() const { return gaus_mean_.size() + pois_obs_.size(); }

double CMSNLL::DoEval(const double* x) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  const unsigned Ng = gaus_mean_.size();
  const unsigned Np = pois_obs_.size();
  double f = 0.;
  for (unsigned i = 0; i < Ng; ++i) {
    const double arg = x[i] - gaus_mean_[i];
    f -= gaus_scale_[i] * arg * arg;
  }
  for (unsigned i = 0; i < Np; ++i) {
    if (x[Ng + i] <= 0.) std::cout << "  x[" << i << "] = " << x[Ng + i] << " " << pois_obs_[i] << "\n";
    f += (-pois_obs_[i] * std::log(x[Ng + i]) + x[Ng + i] + pois_offset_[i]);
  }
  // std::cout << "\n" << (f - zero_point_) << "\n";
  return f - zero_point_;
}

// TODO: Minuit appears not to take advantage of FdF, instead calling DoEval / Grad in sequence
void CMSNLL::Gradient(const double* x, double* grad) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  const unsigned Ng = gaus_mean_.size();
  const unsigned Np = pois_obs_.size();
  for (unsigned i = 0; i < Ng; ++i) {
    const double arg = x[i] - gaus_mean_[i];
    grad[i] = -2. * gaus_scale_[i] * arg;
  }
  for (unsigned i = 0; i < Np; ++i) {
    // const double arg = x[i] - gaus_mean_[i];
    grad[Ng + i] = -pois_obs_[i] / x[Ng + i] + 1.;
  }
}

double CMSNLL::DoDerivative(const double* x, unsigned int icoord) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  std::vector<double> df(NDim());
  Gradient(x, df.data());
  return df[icoord];
}

 
// void CMSNLL::FdF(const double* x, double& f, double* df) const {
//   if (debug) printf(">> %s\n", __FUNCTION__);

// }

void CMSNLL::CheckChanges(const double* x) const {
  unsigned const int N = NDim();
  if (prev_x_.empty()) {
    std::cout << ">> [CheckChanges] First call" << std::endl;
    prev_x_.resize(NDim());
    for (unsigned i = 0; i < N; ++i) {
      prev_x_[i] = x[i];
    }
  } else {
    std::vector<unsigned> changed(NDim());
    changed.resize(0);
    for (unsigned i = 0; i < N; ++i) {
      if (x[i] != prev_x_[i]) {
        changed.push_back(i);
      }
      prev_x_[i] = x[i];
    }
    std::cout << ">> [CheckChanges] [" << changed.size() << "]";
    for (auto const& c : changed) std::cout << " " << c;
    std::cout << std::endl;
  }
}


void CMSNLL::AddGaussianConstraint(double mean, double width) {
  gaus_mean_.push_back(mean);
  gaus_scale_.push_back(-0.5 / (width * width));
}

void CMSNLL::AddPoissonConstraint(double obs) {
  pois_obs_.push_back(obs);
  pois_offset_.push_back(TMath::LnGamma(obs + 1.));
}

std::vector<ROOT::Fit::ParameterSettings> CMSNLL::GetParameters() const {
  TRandom3 rng(12345);
  std::vector<ROOT::Fit::ParameterSettings> res(NDim());
  for (unsigned i = 0; i < gaus_mean_.size(); ++i) {
    res[i].Set(TString::Format("gaus_%i", i).Data(), 0., 1.);
  }
  unsigned offset = gaus_mean_.size();
  for(unsigned i = 0; i < pois_obs_.size(); ++i) {
    res[i + offset].Set(TString::Format("pois_%i", i).Data(), std::max(1., pois_obs_[i] + 1. * rng.Gaus(0, std::sqrt(pois_obs_[i]))), std::sqrt(pois_obs_[i]), 0.001, 200);
  }

  return res;
}

  void CMSNLL::SetZeroPoint(const double *x) {
    zero_point_ = DoEval(x);
    std::cout << ">> Set offset to " << zero_point_ << "\n";
  }

