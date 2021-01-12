//
//  funs.h
//  
//
//  Created by Sameer Deshpande on 8/21/19.
//

#ifndef GUARD_funs_h
#define GUARD_funs_h
#include <RcppArmadillo.h>
#include <cmath>
#include "tree.h"
#include "info.h"
#include <stdio.h>

#endif 


//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo& xi);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree &t, xinfo &xi, tree_prior_info &tree_pi, tree::npv &goodbots);
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars);
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
//double pgrow(tree::tree_p n, xinfo &xi, tree_prior_info &tree_pi);
double pgrow(tree::tree_p n, xinfo &xi, tree_prior_info &tree_pi); // k tells us which beta function we're updating
//--------------------------------------------------
//get sufficients stats for all bottom nodes
void allsuff(tree &x, xinfo &xi, data_info &di, tree::npv &bnv, std::vector<sinfo> &sv);

// get sufficient stats for GROW proposal
void get_suff_grow(sinfo &sl, sinfo &sr, sinfo &sp, tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, data_info &di);

// get sufficient stats for PRUNE proposal
void get_suff_prune(sinfo &sl, sinfo &sr, sinfo &sp, tree &x, tree::tree_p nl, tree::tree_p nr, xinfo &xi, data_info &di);

//--------------------------------------------------


void get_beta_fit(arma::mat &beta, tree &x, xinfo &xi, data_info &di);

void get_fit(std::vector<arma::vec> &phi_beta, arma::mat &tphi_phi_beta, tree &x, xinfo &xi, data_info &di);

void get_fit(arma::mat &beta, std::vector<arma::vec> &phi_beta, arma::mat &tphi_phi_beta, tree &x, xinfo &xi, data_info &di);

/*
void get_beta_fit(std::vector<arma::vec> &beta, tree &x, xinfo &xi, data_info &di);

void get_phi_beta_fit(std::vector<arma::vec> &phi_beta, tree &x, xinfo &xi, data_info &di);

void get_tphi_phi_beta_fit(std::vector<arma::vec> &tphi_phi_beta, tree &x, xinfo &xi, data_info &di);

void get_fit(std::vector<arma::vec> &beta, std::vector<arma::vec> &phi_beta, tree &x, xinfo &xi, data_info &di);
*/
