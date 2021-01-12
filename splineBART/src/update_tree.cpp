//
//  update_tree.cpp

// [SKD]: 4 January 2021
//    Because of improper prior in leaf nodes, restrict grow/prune so that
//    each node has at least 1 subject.

#include "update_tree.h"


void update_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug)
{
  tree::npv goodbots; // leaf notes that are able to be split upon in original tree x
  double PBx = getpb(x, xi, tree_pi, goodbots); // gets the transition prob for growing
  if(debug) Rcpp::Rcout << "  PBx = " << PBx;
  if(gen.uniform() < PBx){
    if(debug) Rcpp::Rcout << "  entering grow_tree" << std::endl;
    grow_tree(x, accept, sigma, theta, var_counts, xi, di, tree_pi, gen, debug);
  } else{
    prune_tree(x, accept, sigma, theta, var_counts, xi, di, tree_pi, gen, debug);
  }
  
  // by this point the decision tree has been updated and we're ready to draw the new jumps
  tree::npv bnv; // vector of pointers to bottom nodes of tree
  std::vector<sinfo> sv; // sufficient statistics for tree
  allsuff(x, xi, di, bnv, sv); // compute the partition cells I for the entire tree
  
  arma::vec m = arma::zeros<arma::vec>(di.D);
  arma::mat P = arma::zeros<arma::mat>(di.D, di.D);
  for(size_t l = 0; l < bnv.size(); l++){
    get_post_params(m, P, sv[l], sigma, di, tree_pi);
    bnv[l]->setm(gen.mvnormal(m,P));
  }
}

// Function to carry out grow move
// goodbots is computed in update_tree
void grow_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug)
{
  tree::npv goodbots; // leaf notes that are able to be split upon in original tree x
  double PBx = getpb(x, xi, tree_pi, goodbots); // gets the transition prob for growing
  
  accept = 0; // initialize indicator of MH acceptance to 0 (reject)
  size_t ni = floor(gen.uniform() * goodbots.size());
  tree::tree_p nx = goodbots[ni]; // nx points to bottom node that is split in the GROW proposal

  std::vector<size_t> goodvars; // variables nx that can non-trivially split on
  getgoodvars(nx, xi, goodvars); // collection of variables that yield non-trivial splits
  // we ultimately don't make use of goodvars
  
  size_t v = gen.multinom(di.p, theta); // randomly pick the variable to split on in the grow proposal
  
  int L,U;
  L = 0;
  U = xi[v].size() - 1;
  nx->rg(v, &L, &U);
  if(U < L){
    // we have a trivial split
    L = 0;
    U = xi[v].size() - 1;
  }
  size_t c = L + floor(gen.uniform() * (U - L + 1));
  
  double Pbotx = 1.0/goodbots.size();
  size_t dnx = nx->depth(); // depth of node nx
  double PGnx = tree_pi.alpha/pow(1.0 + (double) dnx, tree_pi.beta);
  double PGly, PGry;
  if(goodvars.size() > 1){ // we know there are variables we could split l and r on
    PGly = tree_pi.alpha/pow(1.0 + (double) dnx, tree_pi.beta); // depth of new nodes would be 1 + dnx
    PGry = PGly;
  } else{ // only have v to work with, if it is exhausted at either child need PG = 0
    if( (int)(c-1) < L){ // v exhausted in the new left child l, new upper limit would be c-1
      PGly = 0.0;
    } else{
      PGly = tree_pi.alpha/pow(1.0 + 1.0 + (double) dnx, tree_pi.beta);
    }
    if(U < (int)(c+1)){ // v exhausted in new right child, new lower limit would be c+1
      PGry = 0.0;
    } else{
      PGry = tree_pi.alpha/pow(1.0 + 1.0 + (double) dnx, tree_pi.beta);
    }
  }
  double PDy; // prob of proposing death at y
  if(goodbots.size() > 1){ // can birth at y as there are splittable nodes left
    PDy = 1.0 - tree_pi.pb;
  } else{ // nx was the only node we could split on
    if( (PGry == 0) && (PGly == 0)){ // cannot birth at y
      PDy = 1.0;
    } else{ // y can birth at either l or r
      PDy = 1.0 - tree_pi.pb;
    }
  }
  
  double Pnogy; // death prob of choosing the nog node nx in y
  size_t nnogs = x.nnogs();
  tree::tree_cp nxp = nx->getp();
  if(nxp == 0){ // no parent, nx is thetop and only node
    Pnogy = 1.0;
  } else{
    if(nxp->isnog()){ // if parent is a nog, number of nogs same at x and y
      Pnogy = 1.0/nnogs;
    } else{ // if parent is not a nog, y has one more nog.
      Pnogy = 1.0/(nnogs + 1.0);
    }
  }
  
  // get sufficient statistics for proposed left child of nx, proposed right child of nx, and nx
  sinfo sl;
  sinfo sr;
  sinfo sp;
  get_suff_grow(sl, sr, sp, x, nx, v, c, xi, di);
  
  if(sl.n > 0 & sr.n > 0){
    // check that we have at least 1 observation in each of the new leaf nodes
    // otherwise the conditional distribution of (T,mu) is improper
    double lill = compute_lil(sl, sigma, di, tree_pi);
    double lilr = compute_lil(sr, sigma, di, tree_pi);
    double lilp = compute_lil(sp, sigma, di, tree_pi);
    
    double alpha1 = (PGnx*(1.0-PGly)*(1.0-PGry)*PDy*Pnogy)/((1.0-PGnx)*PBx*Pbotx);
    double alpha2 = alpha1 * exp(lill + lilr - lilp + 0.5 * tree_pi.log_det_K - ( (double) di.D - (double) tree_pi.K_ord) * log(tree_pi.tau) + 0.5 * tree_pi.K_ord * log(2.0 * M_PI));
    // remember we pick up a factor of 0.5*logdet(K) - 0.5*log(tau) * (D - order(K)) in each new leaf
    // In GROW move, we have 2 leafs in numerator and 1 in denominator
    // Prior has a factor of (2 pi)^(-(D - K_ord/2)) and when we integrate out mu we get a factor of (2 pi)^(D/2_
    double alpha = std::min(1.0, alpha2);
    if(debug){
    Rcpp::Rcout << "  growing w/ v = " << v << "  c = " << c << std::endl;
    Rcpp::Rcout << "  lill = " << lill << "  lilr = " << lilr << "  lilp = " << lilp << std::endl;
    Rcpp::Rcout << "  alpha1 = " << alpha1 << " alpha2 = " << alpha2  << " alpha = " << alpha << std::endl;
    }
    if(gen.uniform() < alpha){
      // we accept the GROW proposal
      accept = 1;
      var_counts[v]++; // update the running counts of how many times v was selected
      x.birth(nx->nid(),v,c,di.D); // actually perform the birth on x.
    }
  }
  
}

void prune_tree(tree &x, size_t &accept, double &sigma, std::vector<double> &theta, std::vector<size_t> &var_counts, xinfo &xi, data_info &di, tree_prior_info &tree_pi, RNG &gen, bool debug)
{
  tree::npv goodbots; // leaf notes that are able to be split upon in original tree x
  double PBx = getpb(x, xi, tree_pi, goodbots); // gets the transition prob for growing
  tree::npv nognds; // nodes with no grandchildren. hereafter nog nodes
  x.getnogs(nognds);
  size_t ni = floor(gen.uniform() * nognds.size());
  tree::tree_p nx = nognds[ni]; // the nog node we might kill children at
  
  // compute stuff for metropolis-hastings ratio
  double PGny; // prob the nog node grows
  size_t dny = nx->depth();
  PGny = tree_pi.alpha/pow(1.0+dny,tree_pi.beta);
  
  //better way to code these two?
  double PGlx = pgrow(nx->getl(),xi,tree_pi);
  double PGrx = pgrow(nx->getr(),xi,tree_pi);
  
  double PBy;  //prob of birth move at y
               //if(nx->ntype()=='t') { //is the nog node nx the top node
  if(!(nx->p)) { //is the nog node nx the top node
    PBy = 1.0;
  } else {
    PBy = tree_pi.pb;
  }
  
  double Pboty;  //prob of choosing the nog as bot to split on when y
  int ngood = goodbots.size();
  
  tree::tree_p nl = nx->getl();
  tree::tree_p nr = nx->getr();
  
  if(cansplit(nl,xi)) --ngood; //if can split at left child, lose this one
  if(cansplit(nr,xi)) --ngood; //if can split at right child, lose this one
  ++ngood;  //know you can split at nx
  Pboty=1.0/ngood;
  
  double PDx = 1.0-PBx; //prob of a death step at x
  double Pnogx = 1.0/nognds.size();
  
  // what was the split index of nx
  size_t v = nx->v;
  
  sinfo sp;
  sinfo sl;
  sinfo sr;
  
  get_suff_prune(sl, sr, sp, x, nl, nr, xi, di);
  
  double lill = compute_lil(sl, sigma, di, tree_pi);
  double lilr = compute_lil(sr, sigma, di, tree_pi);
  double lilp = compute_lil(sp, sigma, di, tree_pi);

  double alpha1 = ((1.0-PGny)*PBy*Pboty)/(PGny*(1.0-PGlx)*(1.0-PGrx)*PDx*Pnogx);
  double alpha2 = alpha1 * exp(lilp - lilr - lill -0.5 * tree_pi.log_det_K + ( (double) di.D - (double) tree_pi.K_ord) * log(tree_pi.tau) - 0.5 * tree_pi.K_ord * log(2.0 * M_PI));
  // in denominator we have an extra factor of (2pi)^(K_ord/2)

  double alpha = std::min(1.0, alpha2);
  if(debug){
    Rcpp::Rcout << "  lill = " << lill << "  lilr = " << lilr << "  lilp = " << lilp << std::endl;
    Rcpp::Rcout << "  alpha1 = " << alpha1 << "  alpha2 = " << alpha2 << "  alpha = " << alpha << std::endl;
  }
  
  
  if(gen.uniform() < alpha){
    // we accept PRUNE proposal
    accept = 1;
    x.death(nx->nid(),di.D);
    var_counts[v]--; // decrement the variable counts since we have pruned.
  }
}



