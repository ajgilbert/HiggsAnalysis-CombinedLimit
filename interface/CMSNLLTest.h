#ifndef HiggsAnalysis_CombinedLimit_CMSNLLTest
#define HiggsAnalysis_CombinedLimit_CMSNLLTest

#include <sstream>
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

struct Parameter {
  std::string name;
  double value;
  double err = 0.;
  double lo = 0.;
  double hi = 0.;
  Parameter(std::string n, double v, double step) : name(n), value(v), err(step) {}
  Parameter(std::string n, double v, double step, double lo, double hi) : name(n), value(v), err(step), lo(lo), hi(hi) {}
};

struct KappaValue {
  unsigned par;
  double kappa;
  double logKappa;
  KappaValue(unsigned p, double k) : par(p), kappa(k) { logKappa = std::log(kappa); }
};

struct Proc {
  double N0 = 1.;
  std::vector<unsigned> rp;
  // std::vector<unsigned> k;
  // std::vector<double> logkappa;
  std::vector<double> y;
};

struct ProcCache {
  double N = 0.;
  double k_tot = 0.;
  double rp_tot = 0.;
  std::vector<double> dN_rp;
  std::vector<double> dN_rp_work;
  // std::vector<double> dN_k; // = N * logkappa_i
  // std::vector<unsigned> chn_slot; // Where to send the output in the channel dy_rp vector

  void init(Proc const& p);
};

struct Channel {
  // The fixed model data
  unsigned bins = 0;
  unsigned procs = 0;
  std::vector<double> data; // [bins]
  std::vector<Proc> processes; // [procs]
  std::vector<ProcCache> proc_cache; // [procs]
  // std::vector<std::vector<double>> templates; // [procs][bins]
  // std::vector<double> N0; // [procs]
  // std::vector<RateParam> rp_table;

  // Intermediate parts of the calculation
  std::vector<double> y; // [bins]
  std::vector<double> nll_y; // [bins]

  std::vector<double> dy_work;
  std::vector<double> dnll_y_work;

  std::vector<unsigned>  rp_slot; // Where to send the rateParam NLL derivatives [rp]
  std::vector<std::vector<unsigned>> rp_procs;
  std::vector<std::vector<unsigned>> rp_proc_slots;

  std::vector<unsigned> lnN_slot;
  std::vector<std::vector<unsigned>> lnN_procs;
  std::vector<std::vector<double>> lnN_logkappas;

  // std::vector<std::vector<double>> dy_rp; // [rp][bins]
  // std::vector<std::vector<double>> dnll_y_rp; // [rp][bins]
  double nll;
  std::vector<double> dnll_rp; // [rp]


  Channel(unsigned b, unsigned p);
  // void AddLogNormal(unsigned proc, unsigned param, double kappa);
  // void evaluate();
  // inline void SetParameters(std::vector<Parameter>* params) { params_ = params; }

  // std::vector<double> const& nominal();

  // std::vector<std::vector<double>> grad();

};

class CMSNLL : public ROOT::Math::IMultiGradFunction {
private:
  mutable std::vector<Parameter> params_;
  std::map<std::string, unsigned> param_lookup_;
  mutable std::vector<Channel> channels_;

  std::vector<unsigned> gaus_slot_;
  std::vector<double> gaus_mean_;
  std::vector<double> gaus_scale_;
  std::vector<double> pois_obs_;
  std::vector<double> pois_offset_;
  mutable std::vector<double> prev_x_;
  void CheckChanges(const double* x) const;
  double zero_point_ = 0.;

  mutable double nll_; // The cached NLL value
  mutable std::vector<double> dnll_; // The cached derivatives [N params]


public:
  int debug = 0;
  CMSNLL(){};
  ~CMSNLL() override{};
  template<class T>
  std::string FmtVec(std::vector<T> vec, std::string fmt) const;
  ROOT::Math::IMultiGradFunction* Clone() const override;
  double DoEval(const double* x) const override;
  double DoDerivative(const double* x, unsigned int icoord) const override;
  unsigned int NDim() const override;
  void Gradient(const double* x, double* grad) const override;
  // void FdF(const double* x, double& f, double* df) const override;

  void AddParameter(std::string const& name, double val, double step, double lo, double hi);
  double const& val(unsigned const& idx) const;
  double const& val(std::string const& name) const;
  unsigned par(std::string const& name) const;
  // auto kv = KappaValue(0, 1.1);
  //
  unsigned AddChannel(unsigned bins, unsigned procs);

  void SetTemplate(unsigned chn, unsigned proc, std::vector<double> const& x);
  void SetData(unsigned chn, std::vector<double> const& x);

  void PrintModel() const;
  void AddRateParam(unsigned par, unsigned chn, std::vector<unsigned> procs);
  void AddLogNormal(unsigned par, unsigned chn, std::vector<unsigned> const& proc, std::vector<double> const& kappa);
  // void AddLogNormal(std::string const& name, unsigned chn, unsigned proc, double kappa);

  void EvalLogNormal(std::vector<KappaValue> const& kvals, std::vector<double>& result);

  void AddGaussianConstraint(unsigned par, double mean, double width);
  void AddPoissonConstraint(double obs);
  void SetZeroPoint(const double* x);
  void SetParameter(unsigned par, double val);

  double evaluate(bool dograd) const;
  std::vector<ROOT::Fit::ParameterSettings> GetParameters() const;
};


  template<class T>
  std::string CMSNLL::FmtVec(std::vector<T> vec, std::string fmt) const {
    std::stringstream ss;
    ss << "[";
    for (unsigned i = 0; i < vec.size(); ++i) {
      ss << " " << TString::Format(fmt.c_str(), vec[i]);
      if (i < vec.size() - 1) ss << ",";
    }
    ss << "]";
    return ss.str();
  }


#endif