#ifndef HiggsAnalysis_CombinedLimit_CMSNLLTest
#define HiggsAnalysis_CombinedLimit_CMSNLLTest

#include <algorithm>
#include <sstream>
#include <vector>
#include "../interface/Process.h"
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
  Parameter(std::string n, double v, double step, double lo, double hi)
      : name(n), value(v), err(step), lo(lo), hi(hi) {}
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
  std::vector<double> data;           // [bins]
  std::vector<Proc> processes;        // [procs]
  std::vector<ProcCache> proc_cache;  // [procs]
  // std::vector<std::vector<double>> templates; // [procs][bins]
  // std::vector<double> N0; // [procs]
  // std::vector<RateParam> rp_table;

  // Intermediate parts of the calculation
  std::vector<double> y;      // [bins]
  std::vector<double> nll_y;  // [bins]

  std::vector<double> dy_work;
  std::vector<double> dnll_y_work;

  std::vector<unsigned> rp_slot;  // Where to send the rateParam NLL derivatives [rp]
  std::vector<std::vector<unsigned>> rp_procs;
  std::vector<std::vector<unsigned>> rp_proc_slots;

  std::vector<unsigned> lnN_slot;
  std::vector<std::vector<unsigned>> lnN_procs;
  std::vector<std::vector<double>> lnN_logkappas;

  // std::vector<std::vector<double>> dy_rp; // [rp][bins]
  // std::vector<std::vector<double>> dnll_y_rp; // [rp][bins]
  double nll;
  std::vector<double> dnll_rp;  // [rp]

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

  mutable double nll_;                // The cached NLL value
  mutable std::vector<double> dnll_;  // The cached derivatives [N params]

public:
  int debug = 0;
  CMSNLL(){};
  ~CMSNLL() override{};
  template <class T>
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

template <class T>
std::string CMSNLL::FmtVec(std::vector<T> vec, std::string fmt) const {
  std::stringstream ss;
  ss << "[";
  for (unsigned i = 0; i < vec.size(); ++i) {
    ss << " " << TString::Format(fmt.c_str(), vec[i]);
    if (i < vec.size() - 1)
      ss << ",";
  }
  ss << "]";
  return ss.str();
}

class ProcessNorms {
private:
  // Separate into invariant data and cache parts:
  // Invariant data

  struct RateParam {
    std::vector<unsigned> processIndex;
  };

  struct ProcToRateParam {
    std::vector<unsigned> rateParamIndex;
  };


  struct LogNormal {
    std::vector<unsigned> processIndex;
    std::vector<double> logKappa;

    std::vector<unsigned> asymmProcessIndex;
    std::vector<double> logKappaHi;
    std::vector<double> logKappaLo;
  
  };

  struct AsymmLogNormalCache {

    std::vector<double> avg;
    std::vector<double> halfdiff;
    std::vector<double> result;

    void resize(unsigned n) {
      avg.resize(n);
      halfdiff.resize(n);
      result.resize(n);
    }

    unsigned size() const {
      return result.size();
    }
  };

  void logKappaForX(double x, LogNormal const& dat, AsymmLogNormalCache &cache) {
    unsigned const Np = dat.asymmProcessIndex.size();
    if (fabs(x) >= 0.5) {
      if (x >= 0) {
        for (unsigned i = 0; i < Np; ++i) {
          cache.result[i] = dat.logKappaHi[i];
        }
      } else {
        for (unsigned i = 0; i < Np; ++i) {
          cache.result[i] = -dat.logKappaLo[i];
        }
      }
      return;
    } else {
      double twox = x + x;
      double twox2 = twox * twox;
      double alpha = 0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.);
      for (unsigned il = 0; il < Np; ++il) {
        cache.result[il] = cache.avg[il] + alpha * cache.halfdiff[il];
      }
    }
  }

  std::vector<double> N0_;
  std::vector<RateParam> rateParamData_;
  std::vector<LogNormal> logNormalData_;

  std::vector<AsymmLogNormalCache> asymmCache_; // !do not serialize
  

  // This can be generated on the fly, no need to persist
  std::vector<ProcToRateParam> procToRateParamLookup_; // inverse lookup to go from procs to rateParams
  
  std::vector<double> N_; // This is a cached value
  std::vector<double> logValSum_; // This is a cahced value

  std::vector<double> rateParamValues_;
  std::vector<double> logNormalValues_;

  bool init_;
  
public:
  ProcessNorms() {};

  void update(std::vector<double> const& params); // sets new parameter values, nothing else
  void init();
  void sync(); // make N vector up to date, following modes below. mark cache as clean
  /*
  Update modes:
   - FullEval (redo complete evaluation)
   - Memoization (only update pieces that changed) <- should be default?
   - DiffOnly (apply diffs to all the pieces we can)

  Implications:
   - Need to track when caches are in sync or not with current parameter values
   - Need to store "current" parameter values somewhere
  */
  inline std::vector<double> const& Norms() const { return N_; }
  void Grad(unsigned i, std::vector<double> &result); // imply call to sync first
  void Hessian(unsigned i, unsigned j, std::vector<double> &result); // imply call to sync first

};

inline void ProcessNorms::init() {
  N_.resize(N0_.size());
  logValSum_.resize(N0_.size());

  asymmCache_.resize(logNormalData_.size());

  for (unsigned ip = 0; ip < logNormalData_.size(); ++ip) {
    unsigned nAsymm = logNormalData_[ip].asymmProcessIndex.size();
    asymmCache_[ip].resize(nAsymm);
    for (unsigned il = 0; il < nAsymm; ++il) {
      asymmCache_[ip].avg[il] = 0.5*(logNormalData_[ip].logKappaHi[il] - logNormalData_[ip].logKappaLo[il]);
      asymmCache_[ip].halfdiff[il] = 0.5*(logNormalData_[ip].logKappaHi[il] + logNormalData_[ip].logKappaLo[il]);
    }
  }
}

inline void ProcessNorms::Grad(unsigned int i, std::vector<double> &result) {
  for (unsigned ip = 0; ip < N0_.size(); ++ip) {
    result[ip] = 0.;
  }

  /*
    If i is a rateParam:
     Only non-zero grads will be the list of procs this rateParam applies to. Loop through them,
     set N = N0, multiple through by all *other* rateParams and then multiply by exp(logValSum) for that process
  */
  /*
    If i is a log-normal:
    Only non-zero grads will be the lists of procs (sym + asymm) this log-normal applies to. Loop through them,
    Set N' = logKappa * N for sym,
    Set N' = [logkappa + theta * logKappa'] * N for asymm
  */
}

inline void ProcessNorms::Hessian(unsigned int i, unsigned int j, std::vector<double> &result) {
  for (unsigned ip = 0; ip < N0_.size(); ++ip) {
    result[ip] = 0.;
  }

  /*
    If i == j and i is a rateParam:
    Assuming each param can only appear once in the list of rateParams (?), all grads are zero

    If i == j and i is log-normal:
    Set N' = (logKappa^2) * N for sym
    Set N' = [logKappa + theta * logKappa']^2 * N + [logKappa' + logKappa' + theta * logKappa''] * N
  
    If i != j and both are rateParams:
    Only non-zero grads will be list of procs that both rateParams are applied to - so we have to find them

    If i != j and both are log-normal:
    Multiply out the approprate factors as above in Grad

    If i !=j and one rateParam, one log-normal:
    R'i * L + R * L'i
    [R'i'j * L] + [R'i * L'j] + [R'j * L'i] + [R * L'i'j]
  */
}

inline double ProductRuleOneNonZero(std::vector<double> const& f, unsigned nonzero, double df) {
  double res = 1.;
  auto tmp = f;
  tmp[nonzero] = df;
  for (unsigned i = 0; i < f.size(); ++i) {
    res *= tmp[i];
  }
  return res;
}

inline double ProductRule(std::vector<double> const& f, std::vector<double> const& df) {
  unsigned N = f.size();
  std::vector<double> res(N, 1.);
  std::vector<double> work(N, 0.);
  for (unsigned i = 0; i < N; ++i) {
    std::fill(work.begin(), work.end(), f[i]);
    work[i] = df[i];
    for (unsigned j = 0; j < N; ++j) {
      res[j] *= work[j];
    }
  }
  return std::accumulate(res.begin(), res.end(), 0.);
}

inline double ProductRule2(std::vector<double> const& f, std::vector<double> const& dfx, std::vector<double> const& dfy, std::vector<double> const& dfxy) {
  unsigned N = f.size();
  std::vector<double> res(N, 1.);
  // std::vector<double> work(N, 0.);
  // for (unsigned i = 0; i < N; ++i) {
  //     for (unsigned j = 0; j < N; ++j)
  //     std::fill(work.begin(), work.end(), f[i]);
      
  //     work[i] = df[i];
  //     for (unsigned j = 0; j < N; ++j) {
  //       res[j] *= work[j];
  //     }
  //   }
  // }
  return std::accumulate(res.begin(), res.end(), 0.);
}

inline void ProcessNorms::sync() {
  if (!init_) {
    init();
  }

  for (unsigned ip = 0; ip < N0_.size(); ++ip) {
    N_[ip] = N0_[ip];
    logValSum_[ip] = 0.;
  };

  for (unsigned ir = 0; ir < rateParamValues_.size(); ++ir) {
    for (unsigned ipIdx = 0; ipIdx < rateParamData_[ir].processIndex.size(); ++ipIdx) {
      unsigned ip = rateParamData_[ir].processIndex[ipIdx];
      N_[ip] *= rateParamValues_[ir];
    }
  }


  for (unsigned ip = 0; ip < N0_.size(); ++ip) {
    unsigned nRp = procToRateParamLookup_[ip].rateParamIndex.size();
    std::vector<double> f(nRp, 0.);
    std::vector<double> df(nRp, 0.);
    unsigned d_ir = 0; // derive wrt some parameter
    for (unsigned irIdx = 0; irIdx < procToRateParamLookup_[ip].rateParamIndex.size(); ++irIdx) {
      unsigned ir = procToRateParamLookup_[ip].rateParamIndex[irIdx];
      f[irIdx] = rateParamValues_[ir];
      df[irIdx] = (ir == d_ir) ? 1. : 0.;
    }
    ProductRule(f, df);
  }



  for (unsigned il = 0; il < logNormalValues_.size(); ++il) {
    for (unsigned ipIdx = 0; ipIdx < logNormalData_[il].processIndex.size(); ++ipIdx) {
      unsigned ip = logNormalData_[il].processIndex[ipIdx];
      logValSum_[ip] += logNormalValues_[il] * logNormalData_[il].logKappa[ipIdx];
    }
    logKappaForX(logNormalValues_[il], logNormalData_[il], asymmCache_[il]);
    for (unsigned ipIdx = 0; ipIdx < logNormalData_[il].asymmProcessIndex.size(); ++ipIdx) {
      unsigned ip = logNormalData_[il].processIndex[ipIdx];
      logValSum_[ip] += logNormalValues_[il] * asymmCache_[il].result[ipIdx];
    }
  }

  for (unsigned ip = 0; ip < N0_.size(); ++ip) {
    N_[ip] *= std::exp(logValSum_[ip]);
  };

}

class NLLChannel {
  private:
  std::vector<double> data;
  ProcessNorms norms;
  std::vector<double> shapes;
};

#endif
