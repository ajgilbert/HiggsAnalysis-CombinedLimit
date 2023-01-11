#include "../interface/CMSNLLTest.h"
#include <iostream>
#include <numeric>
#include <stdexcept>
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

Channel::Channel(unsigned b, unsigned p) : bins(b), procs(p) {
  data.resize(bins);
  processes.resize(procs);
  proc_cache.resize(procs);

  // std::fill(N0.begin(), N0.end(), 1.);
  // templates.resize(procs);
  // rp_table.resize(procs);
  // lnN_table.resize(procs);
  y.resize(bins);
  nll_y.resize(bins);
  for (unsigned ip = 0; ip < procs; ++ip) {
    processes[ip].y.resize(bins);
  }
}

// void Channel::AddLogNormal(unsigned proc, unsigned param, double kappa) {
//   lnN_table_[proc].push_back(KappaValue{param, kappa});
// }

// void Channel::evaluate() {

  // for (unsigned ip=0; ip < procs_; ++ip) {
  //   norm_cache_[ip] = yield_nominal_[ip];
  //   for (unsigned is = 0; is < lnN_table_[ip].size(); ++is) {
  //     unsigned idx = lnN_table_[ip][is].par;
  //     double scale = std::pow(lnN_table_[ip][is].kappa, (*params_)[idx].value);
  //     std::cout << ">> norm_cache[" << ip << "]: " << (*params_)[idx].name << "[x=" << (*params_)[idx].value << ",kappa=" << lnN_table_[ip][is].kappa << "], scale=" << scale << "\n";
  //     norm_cache_[ip] *= scale;
  //   }
  //   // Here I have to update the process yield modifiers

  // }


ROOT::Math::IMultiGradFunction* CMSNLL::Clone() const { return new CMSNLL(*this); }

void CMSNLL::AddParameter(std::string const& name, double val) {
  if (param_lookup_.count(name)) {
    std::cout << "Error, parameter " << name << " already exists\n";
  } else {
    params_.push_back(Parameter(name, val));
    param_lookup_[name] = params_.size() - 1;
  }
}

double const& CMSNLL::val(unsigned const& idx) const {
  return params_[idx].value;
}

double const& CMSNLL::val(std::string const& name) const {
  return val(par(name));
}

unsigned CMSNLL::par(std::string const& name) const {
  auto it = param_lookup_.find(name);
  if (it == param_lookup_.end()) {
    throw std::runtime_error("Error in par: no parameter with name " + name);
  } else {
    return it->second;
  }  
}

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


unsigned CMSNLL::AddChannel(unsigned bins, unsigned procs) {
  channels_.push_back(Channel(bins, procs));
  return channels_.size() - 1;
  // channels_.back().SetParameters(&params_);
}

void CMSNLL::SetTemplate(unsigned chn, unsigned proc, std::vector<double> const& x) {
  if (x.size() != channels_.at(chn).bins) {
    throw std::runtime_error("Adding template with wrong bin count");
  }
  channels_.at(chn).processes.at(proc).y = x;
}

void CMSNLL::SetData(unsigned chn, std::vector<double> const& x) {
  if (x.size() != channels_.at(chn).bins) {
    throw std::runtime_error("Adding data with wrong bin count");
  }
  channels_.at(chn).data = x;
}



// void CMSNLL::AddLogNormal(std::string const& name, unsigned chn, unsigned proc, double kappa) {
//   if (!param_lookup_.count(name)) {
//     params_.push_back(Parameter{name, 0.});
//     param_lookup_[name] = params_.size() - 1;
//   }
//   unsigned idx = param_lookup_[name];
//   channels_[chn].AddLogNormal(proc, idx, kappa);
// }


void CMSNLL::AddRateParam(unsigned par, unsigned chn, std::vector<unsigned> procs) {
  channels_.at(chn).dy_rp.emplace_back(channels_.at(chn).bins, 0.);
  channels_.at(chn).dnll_y_rp.emplace_back(channels_.at(chn).bins, 0.);
  channels_.at(chn).dnll_rp.emplace_back(0.);
  unsigned slot = channels_.at(chn).dy_rp.size() - 1;
  for (unsigned const& p : procs) {
    channels_.at(chn).processes.at(p).rp.push_back(par);
    channels_.at(chn).proc_cache.at(p).chn_slot.push_back(slot);

  }
  // channels_[chn].rp_table.push_back(RateParam(par, procs)); // TODO: check if already exists
  // channels_[chn].dy_rp.push_back(std::vector<double>(channels_[chn].bins, 0.));
  // channels_[chn].dN_drp.push_back(std::vector<double>(procs.size(), 0.));
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

void CMSNLL::SetParameter(unsigned par, double val) {
  params_[par].value = val;
}


double CMSNLL::evaluate() {
  double ret = 0;
  for (unsigned ic = 0; ic < channels_.size(); ++ic) {
    Channel & chn = channels_[ic];

    // Reset a few things
    // Reset the nominal cache
    std::fill(chn.y.begin(), chn.y.end(), 0.);
    // Reset the rateParam derivative vectors
    for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
      std::fill(chn.dy_rp[ir].begin(), chn.dy_rp[ir].end(), 0.);
    }
    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      Proc & proc = chn.processes[ip];
      ProcCache & pc = chn.proc_cache[ip];
      pc.dN_rp.resize(proc.rp.size()); // Better in some init function
      pc.dN_rp_work.resize(proc.rp.size()); // Better in some init function
      pc.N = chn.processes[ip].N0;
      std::fill(pc.dN_rp.begin(), pc.dN_rp.end(), chn.processes[ip].N0);
      for (unsigned ir = 0; ir < proc.rp.size(); ++ir) {
        // Update the nominal
        double v = val(proc.rp[ir]);
        pc.N *= v;
  
        // Update the derivatives
        std::fill(pc.dN_rp_work.begin(), pc.dN_rp_work.end(), v);
        pc.dN_rp_work[ir] = 1.;
        for (unsigned ir2 = 0; ir2 < proc.rp.size(); ++ir2) {
          pc.dN_rp[ir2] *= pc.dN_rp_work[ir2];
        }
      }
      for (unsigned ir = 0; ir < proc.rp.size(); ++ir) {
        for (unsigned ib = 0; ib < chn.bins; ++ib) {
          chn.dy_rp[pc.chn_slot[ir]][ib] += chn.processes[ip].y[ib] * pc.dN_rp[ir];
        }
      }
      // std::fill(pc.dN_rp.begin(), pc.dN_rp.end(), 0.);
    }

    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      for (unsigned ib = 0; ib < chn.bins; ++ib) {
        chn.y[ib] += chn.processes[ip].y[ib] * chn.proc_cache[ip].N;
      }
  //   for (unsigned is = 0; is < lnN_table_[ip].size(); ++is) {
  //     unsigned idx = lnN_table_[ip][is].par;
  //     double scale = std::pow(lnN_table_[ip][is].kappa, (*params_)[idx].value);
  //     std::cout << ">> norm_cache[" << ip << "]: " << (*params_)[idx].name << "[x=" << (*params_)[idx].value << ",kappa=" << lnN_table_[ip][is].kappa << "], scale=" << scale << "\n";
  //     norm_cache_[ip] *= scale;
  //   }
  //   // Here I have to update the process yield modifiers


    }


    // std::vector<double> perbin_ll(chn.bins, 0.);
    for (unsigned ib = 0; ib < chn.bins; ++ib) {
        chn.nll_y[ib] = chn.data[ib] * std::log(chn.y[ib]) - chn.y[ib];
        std::cout << chn.data[ib] << "\t" << chn.y[ib] << "\t" << chn.nll_y[ib] << "\n";
        for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
          chn.dnll_y_rp[ir][ib] = chn.data[ib] * (chn.dy_rp[ir][ib] / chn.y[ib]) - chn.dy_rp[ir][ib];
        }
    }
    chn.nll = std::accumulate(chn.nll_y.begin(), chn.nll_y.end(), 0.);
    for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
      chn.dnll_rp[ir] = std::accumulate(chn.dnll_y_rp[ir].begin(), chn.dnll_y_rp[ir].end(), 0.);
    }

    ret -= chn.nll;
  }
  return ret;
}

  void CMSNLL::SetZeroPoint(const double *x) {
    zero_point_ = DoEval(x);
    std::cout << ">> Set offset to " << zero_point_ << "\n";
  }

void CMSNLL::PrintModel() {
  auto const& Fmt = TString::Format;
  for (unsigned ic = 0; ic < channels_.size(); ++ic) {
    Channel const& chn = channels_[ic];
    std::cout << ">> Channel " << ic << std::endl;
    std::cout << Fmt("%5s", "Data") << " | " << FmtVec(chn.data, "%5.1f") << std::endl;

    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      std::cout << Fmt("%5i", ip) << " | " << FmtVec(chn.processes[ip].y, "%5.1f") << " | " << Fmt("N = [%5.1f]", chn.processes[ip].N0);
      std::cout << " rp = " << FmtVec(chn.processes[ip].rp, "%2i");
      std::cout << " dN_rp = " << FmtVec(chn.proc_cache[ip].dN_rp, "%5.2f");
      std::cout << " chn_slot = " << FmtVec(chn.proc_cache[ip].chn_slot, "%2i");
      std::cout << std::endl;
    }
    std::cout << Fmt("%5s", "y") << " | " << FmtVec(chn.y, "%5.1f") << std::endl;
    std::cout << "nll_y | " << FmtVec(chn.nll_y, "%5.1f") << " | " << Fmt("nll = [%5.2f]", chn.nll) << std::endl;
    for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
      std::cout << Fmt("%10s", "dy_rp") << "[" << ir << "] | " << FmtVec(chn.dy_rp[ir], "%5.1f") << std::endl;
      std::cout << Fmt("%10s", "dnll_y_rp") << "[" << ir << "] | " << FmtVec(chn.dnll_y_rp[ir], "%5.1f") << " | " << Fmt("dnll_rp = [%5.2f]", chn.dnll_rp[ir]) << std::endl;
    }
  }
}