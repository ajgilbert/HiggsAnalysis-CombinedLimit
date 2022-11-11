#include "../interface/CMSNLLTest.h"
#include <iostream>

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

unsigned int CMSNLL::NDim() const { return gaus_mean_.size(); }

double CMSNLL::DoEval(const double* x) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  double f;
  std::vector<double> df(NDim());
  FdF(x, f, df.data());
  return f;
}

double CMSNLL::DoDerivative(const double* x, unsigned int icoord) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  double f;
  std::vector<double> df(NDim());
  FdF(x, f, df.data());
  return df[icoord];
}

// TODO: Minuit appears not to take advantage of FdF, instead calling DoEval / Grad in sequence
void CMSNLL::Gradient(const double* x, double* grad) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  double f;
  FdF(x, f, grad);
}

void CMSNLL::FdF(const double* x, double& f, double* df) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  CheckChanges(x);
  double res = 0.;
  // TODO: vectorize
  for (unsigned i = 0; i < NDim(); ++i) {
    double arg = x[i] - gaus_mean_[i];
    res -= gaus_scale_[i] * arg * arg;
    df[i] = -2. * gaus_scale_[i] * arg;
  }
  f = res;
}

void CMSNLL::CheckChanges(const double* x) const {
  if (prev_x_.empty()) {
    std::cout << ">> [CheckChanges] First call" << std::endl;
    prev_x_.resize(NDim());
    for (unsigned i = 0; i < NDim(); ++i) {
      prev_x_[i] = x[i];
    }
  } else {
    std::vector<unsigned> changed(NDim());
    changed.resize(0);
    for (unsigned i = 0; i < NDim(); ++i) {
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

std::vector<ROOT::Fit::ParameterSettings> CMSNLL::GetParameters() const {
  std::vector<ROOT::Fit::ParameterSettings> res(NDim());
  for (unsigned i = 0; i < NDim(); ++i) {
    res[i].Set(TString::Format("param_%i", i).Data(), 0., 1.);
  }

  return res;
}
