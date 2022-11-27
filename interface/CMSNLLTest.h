#ifndef HiggsAnalysis_CombinedLimit_CMSNLLTest
#define HiggsAnalysis_CombinedLimit_CMSNLLTest

#include <vector>
#include "TString.h"
#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"
#include "Fit/Fitter.h"

void NearlyEqual(double x1, double x2);

void CheckGrad(ROOT::Math::RichardsonDerivator& d,
               ROOT::Math::Functor1D const& f,
               ROOT::Math::Functor1D const& f_dx,
               double x);

class CMSNLL : public ROOT::Math::IMultiGradFunction {
private:
  std::vector<double> gaus_mean_;
  std::vector<double> gaus_scale_;
  std::vector<double> pois_obs_;
  std::vector<double> pois_offset_;
  double DoEval(const double* x) const override;
  double DoDerivative(const double* x, unsigned int icoord) const override;
  mutable std::vector<double> prev_x_;
  void CheckChanges(const double* x) const;
  double zero_point_ = 0.;
public:
  int debug = 0;
  CMSNLL(){};
  ~CMSNLL() override{};
  ROOT::Math::IMultiGradFunction* Clone() const override;
  unsigned int NDim() const override;
  void Gradient(const double* x, double* grad) const override;
  // void FdF(const double* x, double& f, double* df) const override;
  void AddGaussianConstraint(double mean, double width);
  void AddPoissonConstraint(double obs);
  void SetZeroPoint(const double *x);
  std::vector<ROOT::Fit::ParameterSettings> GetParameters() const;
};

#endif