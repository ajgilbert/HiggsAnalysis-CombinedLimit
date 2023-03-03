#include "../interface/CMSNLLTest.h"
#include <algorithm>
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

  dy_work.resize(bins);
  dnll_y_work.resize(bins);
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

// unsigned int CMSNLL::NDim() const { return gaus_mean_.size() + pois_obs_.size(); }
unsigned int CMSNLL::NDim() const { return params_.size(); }

double CMSNLL::DoEval(const double* x) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  for (unsigned i = 0; i < NDim(); ++i) {
    params_[i].value = x[i];
  }
  evaluate(false);
  return nll_;
  // const unsigned Ng = gaus_mean_.size();
  // const unsigned Np = pois_obs_.size();
  // double f = 0.;
  // for (unsigned i = 0; i < Ng; ++i) {
  //   const double arg = x[i] - gaus_mean_[i];
  //   f -= gaus_scale_[i] * arg * arg;
  // }
  // for (unsigned i = 0; i < Np; ++i) {
  //   if (x[Ng + i] <= 0.) std::cout << "  x[" << i << "] = " << x[Ng + i] << " " << pois_obs_[i] << "\n";
  //   f += (-pois_obs_[i] * std::log(x[Ng + i]) + x[Ng + i] + pois_offset_[i]);
  // }
  // // std::cout << "\n" << (f - zero_point_) << "\n";
  // return f - zero_point_;
}

// TODO: Minuit appears not to take advantage of FdF, instead calling DoEval / Grad in sequence
void CMSNLL::Gradient(const double* x, double* grad) const {
  if (debug) printf(">> %s\n", __FUNCTION__);
  for (unsigned i = 0; i < NDim(); ++i) {
    params_[i].value = x[i];
  }
  evaluate(true);
  // DoEval(x);
  for (unsigned i = 0; i < NDim(); ++i) {
    grad[i] = dnll_[i];

  }

  // for (unsigned i = 0; i < Np; ++i) {
  //   // const double arg = x[i] - gaus_mean_[i];
  //   grad[Ng + i] = -pois_obs_[i] / x[Ng + i] + 1.;
  // }
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
  Channel & channel = channels_.at(chn);
  channel.rp_slot.emplace_back(par);
  channel.dnll_rp.emplace_back(0.);
  // unsigned slot = channels_.at(chn).dy_rp.size() - 1;
  std::vector<unsigned> proc_slots(procs.size(), 0);
  for (unsigned ip = 0; ip < procs.size(); ++ip) {
    Proc & proc = channel.processes.at(procs[ip]);
    proc.rp.push_back(par);
    proc_slots[ip] = proc.rp.size() - 1;
  }
  channel.rp_procs.emplace_back(procs);
  channel.rp_proc_slots.emplace_back(proc_slots);

  // for (unsigned const& p : procs) {
  //   // channels_.at(chn).processes.at(p).rp.push_back(par);
  //   channels_.at(chn).proc_cache.at(p).chn_slot.push_back(slot);

  // }
  // channels_[chn].rp_table.push_back(RateParam(par, procs)); // TODO: check if already exists
  // channels_[chn].dy_rp.push_back(std::vector<double>(channels_[chn].bins, 0.));
  // channels_[chn].dN_drp.push_back(std::vector<double>(procs.size(), 0.));
}

void CMSNLL::AddLogNormal(unsigned par, unsigned chn, std::vector<unsigned> const& proc, std::vector<double> const& kappa) {
  Channel & channel = channels_.at(chn);
  channel.lnN_slot.push_back(par);
  channel.lnN_procs.push_back(proc);
  std::vector<double> logkappa(kappa.size(), 0.);
  std::transform(kappa.begin(), kappa.end(), logkappa.begin(), [](double const& p) { return std::log(p); });
  channel.lnN_logkappas.push_back(logkappa);
}


void CMSNLL::AddGaussianConstraint(unsigned par, double mean, double width) {
  gaus_slot_.push_back(par);
  gaus_mean_.push_back(mean);
  gaus_scale_.push_back(-0.5 / (width * width));
}

void CMSNLL::AddPoissonConstraint(double obs) {
  pois_obs_.push_back(obs);
  pois_offset_.push_back(TMath::LnGamma(obs + 1.));
}

std::vector<ROOT::Fit::ParameterSettings> CMSNLL::GetParameters() const {
  // TRandom3 rng(12345);
  // std::vector<ROOT::Fit::ParameterSettings> res(NDim());
  // for (unsigned i = 0; i < gaus_mean_.size(); ++i) {
  //   res[i].Set(TString::Format("gaus_%i", i).Data(), 0., 1.);
  // }
  // unsigned offset = gaus_mean_.size();
  // for(unsigned i = 0; i < pois_obs_.size(); ++i) {
  //   res[i + offset].Set(TString::Format("pois_%i", i).Data(), std::max(1., pois_obs_[i] + 1. * rng.Gaus(0, std::sqrt(pois_obs_[i]))), std::sqrt(pois_obs_[i]), 0.001, 200);
  // }
  std::vector<ROOT::Fit::ParameterSettings> res(NDim());
  for (unsigned i = 0; i < params_.size(); ++i) {
    // res[i].Set()
    // res[i].Set(params_[i].name, params_[i].value, 1.);
    res[i].Set(params_[i].name, params_[i].value, 2., -10, 20.);
  }

  return res;
}

void CMSNLL::SetParameter(unsigned par, double val) {
  params_[par].value = val;
}


double CMSNLL::evaluate(bool dograd) const {
  // double ret = 0;
  for (unsigned i = 0; i < NDim(); ++i) {
    std::cout << "PARAM " << params_[i].name << " = " << params_[i].value << "\n";
  }
  nll_ = 0.;
  if (dograd) {
    dnll_.resize(params_.size());
    std::fill(dnll_.begin(), dnll_.end(), 0.);
  }

  for (unsigned ic = 0; ic < channels_.size(); ++ic) {
    Channel & chn = channels_[ic];

    // Reset a few things
    // Reset the nominal cache
    std::fill(chn.y.begin(), chn.y.end(), 0.);
    // Reset the rateParam derivative vectors
    // if (dograd) {
    //   std::fill(chn.dy_work.begin(), chn.dy_work.end(), 0.);
    //   }
    // }
    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      Proc & proc = chn.processes[ip];
      ProcCache & pc = chn.proc_cache[ip];
      pc.N = chn.processes[ip].N0;
      pc.k_tot = 0.; // reset the log-normal sum
      if (dograd) {
        pc.dN_rp.resize(proc.rp.size()); // Better in some init function
        pc.dN_rp_work.resize(proc.rp.size()); // Better in some init function
        std::fill(pc.dN_rp.begin(), pc.dN_rp.end(), chn.processes[ip].N0);
      }
      for (unsigned ir = 0; ir < proc.rp.size(); ++ir) {
        // Update the nominal
        double v = val(proc.rp[ir]);
        pc.N *= v;
  
        // Update the derivatives
        if (dograd) {
          std::fill(pc.dN_rp_work.begin(), pc.dN_rp_work.end(), v);
          pc.dN_rp_work[ir] = 1.;
          for (unsigned ir2 = 0; ir2 < proc.rp.size(); ++ir2) {
            pc.dN_rp[ir2] *= pc.dN_rp_work[ir2];
          }
        }
      }
      // if (dograd) {
      //   for (unsigned ir = 0; ir < proc.rp.size(); ++ir) {
      //     for (unsigned ib = 0; ib < chn.bins; ++ib) {
      //       chn.dy_rp[pc.chn_slot[ir]][ib] += chn.processes[ip].y[ib] * pc.dN_rp[ir];
      //     }
      //   }
      // }
      // std::fill(pc.dN_rp.begin(), pc.dN_rp.end(), 0.);
    }

    for (unsigned ik = 0; ik < chn.lnN_slot.size(); ++ik) {
      for (unsigned ip = 0; ip < chn.lnN_procs[ik].size(); ++ip) {
        chn.proc_cache[chn.lnN_procs[ik][ip]].k_tot += val(chn.lnN_slot[ik]) * chn.lnN_logkappas[ik][ip];
      }
    }
    
    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      chn.proc_cache[ip].N *= std::exp(chn.proc_cache[ip].k_tot);
      for (unsigned ib = 0; ib < chn.bins; ++ib) {
        chn.y[ib] += chn.processes[ip].y[ib] * chn.proc_cache[ip].N;
      }
    }


    for (unsigned ib = 0; ib < chn.bins; ++ib) {
        chn.nll_y[ib] = chn.data[ib] * (std::log(chn.y[ib]) - std::log(chn.data[ib])) - chn.y[ib] + chn.data[ib];
        // std::cout << chn.data[ib] << "\t" << chn.y[ib] << "\t" << chn.nll_y[ib] << "\n";
        // if (dograd) {
        //   for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
        //     chn.dnll_y_rp[ir][ib] = chn.data[ib] * (chn.dy_rp[ir][ib] / chn.y[ib]) - chn.dy_rp[ir][ib];
        //   }
        // }
    }
    chn.nll = std::accumulate(chn.nll_y.begin(), chn.nll_y.end(), 0.);
    nll_ -= chn.nll;

    if (dograd) {
      for (unsigned ir = 0; ir < chn.rp_slot.size(); ++ir) {
        std::fill(chn.dy_work.begin(), chn.dy_work.end(), 0.);
        std::fill(chn.dnll_y_work.begin(), chn.dnll_y_work.end(), 0.);
        for (unsigned ip = 0; ip < chn.rp_procs[ir].size(); ++ ip) {
          double const& dN_rp = chn.proc_cache[chn.rp_procs[ir][ip]].dN_rp[chn.rp_proc_slots[ir][ip]];
          for (unsigned ib = 0; ib < chn.bins; ++ib) {
            chn.dy_work[ib] += chn.processes[chn.rp_procs[ir][ip]].y[ib] * dN_rp;
          }
        }
        for (unsigned ib = 0; ib < chn.bins; ++ib) {
          chn.dnll_y_work[ib] = chn.data[ib] * (chn.dy_work[ib] / chn.y[ib]) - chn.dy_work[ib];
        }
        chn.dnll_rp[ir] = std::accumulate(chn.dnll_y_work.begin(), chn.dnll_y_work.end(), 0.);
        dnll_[chn.rp_slot[ir]] -= chn.dnll_rp[ir];
      }

      for (unsigned ik = 0; ik < chn.lnN_slot.size(); ++ik) {
        std::fill(chn.dy_work.begin(), chn.dy_work.end(), 0.);
        std::fill(chn.dnll_y_work.begin(), chn.dnll_y_work.end(), 0.);
        for (unsigned ip = 0; ip < chn.lnN_procs[ik].size(); ++ip) {
          double dN_k = chn.proc_cache[chn.lnN_procs[ik][ip]].N * chn.lnN_logkappas[ik][ip];
          for (unsigned ib = 0; ib < chn.bins; ++ib) {
            chn.dy_work[ib] += chn.processes[chn.lnN_procs[ik][ip]].y[ib] * dN_k;
          }
        }
        for (unsigned ib = 0; ib < chn.bins; ++ib) {
          chn.dnll_y_work[ib] = chn.data[ib] * (chn.dy_work[ib] / chn.y[ib]) - chn.dy_work[ib];
        }
        // chn.dnll_rp[ir] = std::accumulate(chn.dnll_y_work.begin(), chn.dnll_y_work.end(), 0.);
        dnll_[chn.lnN_slot[ik]] -= std::accumulate(chn.dnll_y_work.begin(), chn.dnll_y_work.end(), 0.);
      }
    }
    // if (dograd) {
    //   for (unsigned ir = 0; ir < chn.dy_rp.size(); ++ir) {
    //     chn.dnll_rp[ir] = std::accumulate(chn.dnll_y_rp[ir].begin(), chn.dnll_y_rp[ir].end(), 0.);
    //     dnll_[chn.rp_slot[ir]] -= chn.dnll_rp[ir];
    //   }
    // }
  } // channels
  
  // Do constraint part
  for (unsigned i = 0; i < gaus_mean_.size(); ++i) {
    const double arg = val(gaus_slot_[i]) - gaus_mean_[i];
    nll_ -= gaus_scale_[i] * arg * arg;
    // grad[i] = -2. * gaus_scale_[i] * arg;
  }
  if (dograd) {
    for (unsigned i = 0; i < gaus_mean_.size(); ++i) {
      const double arg = val(gaus_slot_[i]) - gaus_mean_[i];
      dnll_[gaus_slot_[i]] -= 2. * gaus_scale_[i] * arg;
    }
  }
  // this->PrintModel();
  return 0.;
}

  void CMSNLL::SetZeroPoint(const double *x) {
    zero_point_ = DoEval(x);
    std::cout << ">> Set offset to " << zero_point_ << "\n";
  }

void CMSNLL::PrintModel() const {
  auto const& Fmt = TString::Format;
  for (unsigned ic = 0; ic < channels_.size(); ++ic) {
    Channel const& chn = channels_[ic];
    std::cout << ">> Channel " << ic << std::endl;
    std::cout << Fmt("%5s", "Data") << " | " << FmtVec(chn.data, "%5.1f") << std::endl;

    for (unsigned ip = 0; ip < chn.procs; ++ip) {
      std::cout << Fmt("%5i", ip) << " | " << FmtVec(chn.processes[ip].y, "%5.1f") << " | " << Fmt("N = [%5.1f]", chn.processes[ip].N0);
      std::cout << " rp = " << FmtVec(chn.processes[ip].rp, "%2i");
      std::cout << " dN_rp = " << FmtVec(chn.proc_cache[ip].dN_rp, "%5.2f");
      // std::cout << " chn_slot = " << FmtVec(chn.proc_cache[ip].chn_slot, "%2i");
      std::cout << std::endl;
    }
    std::cout << Fmt("%5s", "y") << " | " << FmtVec(chn.y, "%5.1f") << std::endl;
    std::cout << "nll_y | " << FmtVec(chn.nll_y, "%5.1f") << " | " << Fmt("nll = [%5.2f]", chn.nll) << std::endl;
    for (unsigned ir = 0; ir < chn.rp_slot.size(); ++ir) {
      // std::cout << Fmt("%10s", "dy_rp") << "[" << ir << "] | " << FmtVec(chn.dy_rp[ir], "%5.1f") << std::endl;
      std::cout << Fmt("dnll_rp = [%5.2f]", chn.dnll_rp[ir]) << Fmt(", rp_slot = [%i]", chn.rp_slot[ir]) << std::endl;
    }
  }
  std::cout << ">> Totals:" << std::endl;
  std::cout << Fmt("nll = [%7.3f]", nll_) << std::endl;
  std::cout << "dnll | " << FmtVec(dnll_, "%6.2f") << "|" << std::endl;
}