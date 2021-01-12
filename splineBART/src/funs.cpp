//
//  funs.cpp
//  

#include "funs.h"

//--------------------------------------------------
//does a (bottom) node have variables you can split on?
bool cansplit(tree::tree_p n, xinfo& xi)
{
  int L,U;
  bool v_found = false; //have you found a variable you can split on
  size_t v=0;
  while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
    L=0; U = xi[v].size()-1;
    n->rg(v,&L,&U);
    if(U>=L) v_found=true;
    v++;
  }
  return v_found;
}
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree &t, xinfo &xi, tree_prior_info &tree_pi, tree::npv &goodbots){
  double pb; // prob of birth to be returned
  tree::npv bnv; // all the bottom nodes
  t.getbots(bnv); // actually find all of the bottom nodes
  for(size_t i = 0; i != bnv.size(); i++){
    if(cansplit(bnv[i], xi)) goodbots.push_back(bnv[i]);
  }
  if(goodbots.size() == 0) pb = 0.0; // there are no bottom nodes you can split on
  else{
    if(t.treesize() == 1) pb = 1.0; // tree only has one node
    else pb = tree_pi.pb;
  }
  return pb;
}
//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
  int L,U;
  for(size_t v=0;v!=xi.size();v++) {//try each variable
    L=0; U = xi[v].size()-1;
    n->rg(v,&L,&U);
    if(U>=L) goodvars.push_back(v);
  }
}
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo &xi, tree_prior_info &tree_pi)
{
  if(cansplit(n,xi)) return tree_pi.alpha/pow(1.0 + n->depth(), tree_pi.beta);
  else return 0.0;
}
//--------------------------------------------------
//get sufficients stats for all bottom nodes

void allsuff(tree &x, xinfo &xi, data_info &di, tree::npv &bnv, std::vector<sinfo> &sv){
  
  // Bottom nodes are written to bn.
  // Sufficient statistics are written to elements of sv (each element is of class sinfo)
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni; //the  index into vector of the current bottom node
  double *zz; //current z
  bnv.clear(); // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv); // Save bottom nodes for x to bnv variable.
  
  typedef tree::npv::size_type bvsz;  // Is a better C way to set type.  (tree::npv::size_type) will resolve to an integer,
                                      // or long int, etc.  We don't have to know that ahead of time by using this notation.
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  
  sv.clear();
  sv.resize(nb);
  
  // this may be unnecessary
  for(size_t l = 0; l != bnv.size(); l++){
    sv[l].n = 0;
    sv[l].I.clear(); // clear out I set for each node
  }
  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  // [SKD] 27 Aug 2019 -- I changed the loop to iterate over l for consistenct with what's above
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz l=0;l!=bnv.size();l++) bnmap[bnv[l]]=l;
  // bnv[l]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2
  // Loop through each observation.  Push each obs x down the tree and find its bottom node,
  // then index into the suff stat for the bottom node corresponding to that obs.
  
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i*di.p; // Now points to beginning of covariates for surface i.
    tbn = x.bn(zz,xi); // Finds bottom node for surface i
    ni = bnmap[tbn]; // Maps bottom node to integer index
    ++(sv[ni].n); // increment total count (across individuals)
    sv[ni].I.push_back(i); // add i to the list of surfaces which land in this node
  }
}

void get_suff_grow(sinfo &sl, sinfo &sr, sinfo &sp, tree &x, tree::tree_cp nx, size_t v, size_t c, xinfo &xi, data_info &di)
{
  double* zz;
  sl.n = 0;
  sl.I.clear();
  
  sr.n = 0;
  sr.I.clear();
  
  sp.n = 0;
  sp.I.clear();
  
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p; // now points to beginning of surface i's covariates
    if(nx == x.bn(zz,xi)){
      if(zz[v] < xi[v][c]){
        ++(sl.n);
        sl.I.push_back(i);
      } else{
        ++(sr.n);
        sr.I.push_back(i);
      }
      ++(sp.n);
      sp.I.push_back(i);
    }
  }
}

void get_suff_prune(sinfo &sl, sinfo &sr, sinfo &sp, tree &x, tree::tree_p nl, tree::tree_p nr, xinfo &xi, data_info &di)
{
  double* zz;
  sl.n = 0;
  sl.I.clear();
  
  sr.n = 0;
  sr.I.clear();
  
  sp.n = 0;
  sp.I.clear();
  
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    tree::tree_cp bn = x.bn(zz,xi);
    if(bn == nl){
      // surface i goes to nl (the left child)
      ++(sl.n);
      sl.I.push_back(i);
      ++(sp.n);
      sp.I.push_back(i);
    } else if(bn == nr){
      ++(sr.n);
      sr.I.push_back(i);
      ++(sp.n);
      sp.I.push_back(i);
    }
  }
}

// only used when we are saving training and testing beta's
void get_beta_fit(arma::mat &beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn; // bottom node pointer
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi); // get bottom node for observation i
    beta.col(i) = bn->getm();
  }
}

// this is used in the main backfitting loop
void get_fit(std::vector<arma::vec> &phi_beta, arma::mat &tphi_phi_beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    phi_beta[i] = di.phi->at(i) * bn->getm();
    tphi_phi_beta.col(i) = di.tphi_phi->at(i) * bn->getm();
  }
}

// overloaded version that also returns mu, Phi * mu, and t(Phi) * Phi * mu.
// probably never called
void get_fit(arma::mat &beta, std::vector<arma::vec> &phi_beta, arma::mat &tphi_phi_beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    beta.col(i) = bn->getm();
    phi_beta[i] = di.phi->at(i) * bn->getm();
    tphi_phi_beta.col(i) = di.tphi_phi->at(i) * bn->getm();
  }
}

//--------------------------------------------------


/*
 main loop will go something like this:
 
 get_fit(beta, phi_beta, tphi_phi_beta, t_vec[m], xi, di)
 for(size_t i = 0; i < n; i++){
   // remove fit of tree m from residual.
   // r = y - Phi * (beta_(-m) + beta_m)
   rf[i] += phi_beta[i];
   tphi_rf[i] += tphi_phi_beta.col(i);
 
 
 }
 
 
 
 fit(t_vec[m], xi, di, tree_eval)
 // tree_eval[i] is a D x 1 vector holding g(z_i; T_m, mu_m)
 for(size_t i = 0; i < n; i++){
   // currently di.phi_beta[i] is Phi_i * beta
   // we need to remove fit of tree T_m from beta
   di.phi_beta[i] -= phi[i] * tree_eval[i];
   // currently di.tphi_r[i] is t(Phi[i]) * (y_i - Phi_i * beta)
   // we need to temporarily remove fit of tree T_m from this
   di.tphi_r[i] += tphi_phi[i] * tree_eval[i];
 }
 
 update_tree
 
 fit(t_vec[m], xi, di, tree_eval)
 for(size_t i = 0; i < n; i++){
   // currently di.phi_beta[i] is missing the contribution from tree m
   di.phi_beta[i] += di.phi[i] * tree_eval[i];
   di.tphi_r[i] -= di.tphi_phi[i] * tree_eval[i];
 }
 
 After all trees are update di.phi_beta[i] has the new value of
 
 */

/*
void get_beta_fit(std::vector<arma::vec> &beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    beta[i] = bn->getm();
  }
}

void get_phi_beta_fit(std::vector<arma::vec> &phi_beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    phi_beta[i] = di.phi->at(i) * bn->getm();
  }
}

void get_tphi_phi_beta_fit(std::vector<arma::vec> &tphi_phi_beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    tphi_phi_beta[i] = di.tphi_phi->at(i) * bn->getm();
  }

}

void get_fit(std::vector<arma::vec> &beta, std::vector<arma::vec> &phi_beta, tree &x, xinfo &xi, data_info &di)
{
  double *zz;
  arma::vec tmp_beta;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.n; i++){
    zz = di.z + i * di.p;
    bn = x.bn(zz,xi);
    tmp_beta = bn->getm();
    beta[i] = tmp_beta;
    phi_beta[i] = di.phi->at(i) * tmp_beta;
  }
}
*/
/*
void fit(tree &x, xinfo &xi, data_info &di, std::vector<arma::mat> &phi_vec, double* beta, double *phi_beta)
{
  arma::vec tmp_beta;
  arma::vec tmp_phi_beta;
  double *zz;
  tree::tree_cp bn;
  for(size_t i = 0; i < di.N; i++){
    zz = di.z + di.p*i;
    bn = t.bn(zz,xi);
    tmp_beta = bn->getm(); // pulled the correctly sized beta
    for(size_t d = 0; d < di.D; d++) beta[d + di.D*i] = tmp_beta(d);
    tmp_phi_beta = phi_vec[i] * tmp_beta;
    for(size_t t = 0; t < di.n[i]; t++) phi_beta[t + di.start_index[i]] = tmp_phi_beta(t);
  }
}

// phi_beta holds the fit of a single Phi * g(z, T, mu)
// eventually use that to update allfit
*/
