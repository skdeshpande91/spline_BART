//
//  test_fit.cpp
//

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <vector>
#include <stdio.h>
#include "info.h"
#include "rng.h"
#include "funs.h"
#include "draw_tree.h"

// [[Rcpp::export]]
Rcpp::List test_tree_fit(Rcpp::List Y_train,
                         Rcpp::List Phi_train,
                         arma::mat Z_train,
                         Rcpp::List cutpoints,
                         size_t D)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  
  size_t n_train = Z_train.n_rows;
  size_t p = Z_train.n_cols;
  
  std::vector<arma::mat> phi_train(n_train);
  std::vector<arma::vec> rf(n_train);
  
  std::vector<arma::mat> tphi_phi(n_train);
  std::vector<arma::vec> tphi_rf(n_train);
  
  double* z_train_ptr = new double[n_train * p];
  
  size_t N = 0; // total number of observations
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    
    Rcpp::NumericVector tmp_y = Y_train[i];
    arma::vec tmp_y2(tmp_y.begin(), tmp_y.size(), false, true);
    
    phi_train[i] = tmp_phi2;
    rf[i] = tmp_y2;
    
    // compute t(Phi) * y and t(Phi) * Phi only once and do it here to simplify input
    
    tphi_rf[i] = tmp_phi2.t() * tmp_y2;
    tphi_phi[i] = tmp_phi2.t() * tmp_phi2;
    
    N += rf[i].size();
    
    for(size_t j = 0; j < p; j++) z_train_ptr[j + p*i] = Z_train(i,j);
  }
  Rcpp::Rcout << "  Total of " << N << " observations from " << n_train << " subjects." << std::endl;

  
  // Read in and format the cutpoints
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = cutpoints[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  
  // create data object
  data_info di;
  di.n = n_train;
  di.N = N;
  di.p = p;
  di.D = D;
  di.z = &z_train_ptr[0];
  di.phi = &phi_train;
  di.rf = &rf;
  di.tphi_phi = &tphi_phi;
  di.tphi_rf = &tphi_rf;
  
  // create tree_prior object
  tree_prior_info tree_pi; // defaults to the CGM10 prior on the tree
  
  
  // begin draw tree
  tree t;
  tree::npv bnv;
  std::vector<sinfo> sv;
  bool empty_leaf = true;
  size_t counter = 0;
  size_t L = 0;
  // for testing, when we randomly generate tree
  // we need to ensure all leafs have at least one element
  while(empty_leaf && (counter < 5000)){
    empty_leaf = false;
    draw_tree(t, xi, tree_pi.alpha, tree_pi.beta, D, gen);
    allsuff(t, xi, di, bnv, sv);
    L = bnv.size();
    for(size_t l = 0; l < L; l++){
      if(sv[l].n == 0){
        empty_leaf = true;
        break;
      }
    }
    counter++;
  }
  // end draw tree
  if(!empty_leaf){
    Rcpp::Rcout << "Took " << counter << " tries to get tree with non-empty leaves" << std::endl;
    t.print(true);
    
    arma::mat unik_beta = arma::zeros<arma::mat>(D,L);
    Rcpp::NumericVector leaf_ix(n_train);
    for(size_t l = 0; l < L; l++){
      bnv[l]->setm(gen.std_norm_vec(D)); // set beta for this random tree
      unik_beta.col(l) = bnv[l]->getm();
      // record the index of the leaf to which each observation goes
      // remember getbots returns leafs in left-to-right order
      for(size_t ix = 0; ix < sv[l].n; ix++) leaf_ix[sv[l].I[ix]] = l+1; // R is 1-indexed!
    }
    
    // now that the tree is fully populated, we can compute the fit and
    
    arma::mat beta = arma::zeros<arma::mat>(D, n_train);
    std::vector<arma::vec> phi_beta(n_train);
    arma::mat tphi_phi_beta = arma::zeros<arma::mat>(D, n_train);
    
    get_fit(beta, phi_beta, tphi_phi_beta, t, xi, di);
    
    // now prepare the output container
    Rcpp::List phi_beta_out(n_train);
    for(size_t i = 0; i < n_train; i++) phi_beta_out[i] = phi_beta[i];
    
    Rcpp::List results;
    results["unik_beta"] = unik_beta;
    results["leaf_ix"] = leaf_ix;
    results["beta"] = beta;
    results["phi_beta"] = phi_beta_out;
    results["tphi_phi_beta"] = tphi_phi_beta;
    return(results);
  } else{
    Rcpp::Rcout << "  counter  = " << counter << std::endl;
    Rcpp::stop("after 5000 tries, tree has empty leaves.");
  }
}
/*
 old version
 // old test_fit. look at it later
 tree::npv bnv;
 t.getbots(bnv); // returns node in left - to - right order, regardless of depth
 for(size_t l = 0; l < bnv.size(); l++){
 bnv[l]->setm(gen.std_norm_vec(D));
 }
 // now get the fit of this tree
 
 std::vector<arma::vec> beta_train(n_train);
 std::vector<arma::vec> phi_beta_train(n_train);
 
 get_fit(beta_train, phi_beta_train, t, xi, di);
 Rcpp::Rcout << "got the fit!" << std::endl;
 
 
 Rcpp::Rcout << "tree has " << bnv.size() << " bottom nodes" << std::endl;
 arma::mat unik_beta = arma::zeros<arma::mat>(D, bnv.size());
 Rcpp::Rcout << "bottom node ids:";
 for(size_t l = 0; l < bnv.size(); l++){
 Rcpp::Rcout << " " << bnv[l]->nid();
 unik_beta.col(l) = bnv[l]->getm();
 }
 Rcpp::Rcout << std::endl;
 
 // output containers
 
 arma::mat beta_out = arma::zeros<arma::mat>(D, n_train);
 for(size_t i = 0; i < n_train; i++) beta_out.col(i) = beta_train[i];
 
 Rcpp::List phi_beta_out(n_train);
 for(size_t i = 0; i < n_train; i++) phi_beta_out[i] = phi_beta_train[i];
 
 
 // In order to test that our fit function is correct, we will save the bottom node betas
 // Then based on the printed output of the tree, we can build the partition in R
 // and then write a little loop that looks at which bottom node we're in and what the corresponding beta is
 // finally compare that to
 
 
 //Rcpp::List phi_beta_out;
 
 //Rcpp::Rcout << "Finished printing tree" << std::endl;
 
 
 
 //for(size_t i = 0; i < n_train; i++) phi_beta_out[i] = arma::zeros<arma::vec>(phi_train[i].n_rows);
 
 Rcpp::List results;
 results["beta"] = beta_out;
 results["phi_beta"] = phi_beta_out;
 results["unik_beta"] = unik_beta;
 return(results);
 
 */


/*
// [[Rcpp::export]]
Rcpp::List test_tree_init(Rcpp::List Phi_train,
                          Rcpp::List tPhi_Phi_train,
                          Rcpp::List tPhi_Phi_train,
                          arma::mat Z_train,
                          Rcpp::List Phi_test,
                          arma::mat Z_test,
                          size_t D, // number of basis elements
                          size_t K_ord, // rank deficiency of precision matrix (usually 1 or 2)
                          arma::mat K, // prior precision matrix in each leaf
                          double log_det_K, // generalized log determinant of prior precision
                          double alpha, double beta,
                          size_t M)
{
  Rcpp::RNGScope scope;
  RNG gen;
  
  
  size_t n_train = Z_train.n_rows;
  size_t n_test = Z_test.n_rows;
  size_t p = Z_train.n_cols;
  
  std::vector<arma::mat> phi_train(n_train);
  std::vector<arma::mat> phi_test(n_test);
  
  std::vector<arma::mat> tphi_phi(n_train);
  std::vector<arma::vec> tphi_rf(n_train);
  
  
  
  double* z_train_ptr = new double[n_train * p];
  double* z_test_ptr = new double[n_test * p];
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    
    Rcpp::NumericMatrix tmp_tphi_phi = tPhi_Phi_train[i];
    arma::mat tmp_tphi_phi2(tmp_tphi_phi.begin(), tmp_tphi_phi.rows(), tmp_tphi_phi.cols(), false, true);
    
    Rcpp::NumericVector tmp_tphi_y = tPhi_Y_train[i];
    arma::vec tmp_tphi_y2(tmp_tphi_y.begin(), tmp_tphi_y.size(), false, true);
    
    phi_train[i] = tmp_phi2;
    tphi_phi[i] = tmp_tphi_phi2;
    tphi_rf[i] = tmp_tphi_y2;
    
    for(size_t j = 0; j < p; j++) z_train_ptr[j + p*i] = Z_train(i,j);
  }
  
  for(size_t i = 0; i < n_test; i++){
    Rcpp::NumericVector tmp_phi = Phi_test[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    phi_test[i] = tmp_phi2;
    for(size_t j = 0; j < p; j++) z_test_ptr[j + p*i] = Z_test(i,j);
  }
  
  
  // Read in and format the cutpoints
  xinfo xi;
  xi.resize(p);
  for(size_t j = 0; j < p; j++){
    Rcpp::NumericVector tmp = xinfo_list[j];
    std::vector<double> tmp2;
    for(int jj = 0; jj < tmp.size(); jj++) tmp2.push_back(tmp[jj]);
    xi[j] = tmp2;
  }
  
  std::vector<tree> t_vec(M); // create a vector of M trees
  std::vector<double> theta(p, 1.0/( (double) p)); // fixed vector of splitting probabilities
  std::vector<size_t> var_counts(p, 0);
  
  // initialize all of the trees to start with the 0 vector in their leaves
  for(size_t m = 0; m < M; m++){
    t_vec[m].setm(arma::zeros<vec>(D));
  }
  
  // initialize vectors holding overall fits (i.e. the beta's and values of Phi * beta)
  
  std::vector<arma::mat> tphi_phi_fit(n_train); // holds Phi_i' * Phi_i * g(z;T_m).
  // constantly evaluates this in back-fitting and subtract/add it to tphi_rf.
  
  std::vector<arma::vec> beta_test(n_test); // saves the value of spline coefficients for test set
  std::vector<arma::vec> phi_beta_test(n_test); // saves value of Phi * beta for test set
  
  for(size_t i = 0; i < n_train; i++){
    tphi_phi_fit[i] = arma::zeros<arma::vec>(phi[i].n_rows()); // initialize temporary fit to n_i x 1 vector
  }
  
  for(size_t i = 0; i < n_test; i++){
    
  }
  
  
  
}

*/



/*

// [[Rcpp::export]]
Rcpp::List test_input(Rcpp::List Phi_train,
                      Rcpp::List tPhi_Phi_train,
                      Rcpp::List tPhi_Y_train,
                      arma::mat Z_train,
                      size_t D)
{
  
  size_t n_train = Z_train.n_rows;
  size_t p = Z_train.n_cols;
  
  std::vector<arma::mat> phi(n_train);
  std::vector<arma::mat> tphi_phi(n_train);
  std::vector<arma::vec> tphi_rf(n_train);
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat tmp_phi2(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    
    Rcpp::NumericMatrix tmp_tphi_phi = tPhi_Phi_train[i];
    arma::mat tmp_tphi_phi2(tmp_tphi_phi.begin(), tmp_tphi_phi.rows(), tmp_tphi_phi.cols(), false, true);
    
    Rcpp::NumericVector tmp_tphi_y = tPhi_Y_train[i];
    arma::vec tmp_tphi_y2(tmp_tphi_y.begin(), tmp_tphi_y.size(), false, true);
    
    phi[i] = tmp_phi2;
    tphi_phi[i] = tmp_tphi_phi2;
    tphi_rf[i] = tmp_tphi_y2;
  }
  
  
  data_info di;
  di.n = n_train;
  di.p = p;
  di.D = D;
  di.phi = &phi;
  di.tphi_phi = &tphi_phi;
  di.tphi_rf = &tphi_rf;
  
  Rcpp::Rcout << "Before updating tphi_rf[0]" << std::endl;
  di.tphi_rf->at(0).print();
  // see here for syntax: https://stackoverflow.com/questions/6946217/how-to-access-the-contents-of-a-vector-from-a-pointer-to-the-vector-in-c
  
  size_t ni = tphi_rf[0].size();
  
  tphi_rf[0] += arma::ones<arma::vec>(ni);
  
  Rcpp::Rcout << "After update tphi_rf[0]" << std::endl;
  di.tphi_rf->at(0).print();
  
  Rcpp::List results;
  results["phi0"] = phi[0];
  return(results);
}
*/

/*
// [[Rcpp::export]]
Rcpp::List test_pointer(Rcpp::List Phi_train,
                        Rcpp::List tPhiY_train,
                        arma::mat Z_train)
{
  
  size_t n_train = Z_train.n_rows;



  std::vector<arma::mat>* phi_ptr = new std::vector<arma::mat>();
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericMatrix tmp_phi = Phi_train[i];
    arma::mat phi(tmp_phi.begin(), tmp_phi.rows(), tmp_phi.cols(), false, true);
    phi_ptr->push_back(phi);
  }
  arma::mat phi0 = phi_ptr->at(0);
  arma::mat phi1 = (*phi_ptr)[1];
  
  
  Rcpp::List results;
  results["phi0"] = phi[0];
  results["phi1"] = phi[1];
  return(results);
  
}
*/



/*
Rcpp::List test_spline(Rcpp::List Phi_train,
                       Rcpp::List tPhi_Y,
                       arma::mat Z_train,
                       Rcpp::List Phi_test,
                       Rcpp::mat Z_test,
                       Rcpp::List xinfo_list,
                       arma::mat Omega,
                       double log_det_Omega,
                       size_t D,
                       size_t M,
                       size_t burn, size_t nd,
                       bool verbose, size_t print_every,
                       double a, double b, size_t N_u, double rho_alpha,
                       double nu_sigma)
{
  if(verbose == true) Rcpp::Rcout << "Entering splineBART" << std::endl;
  Rcpp::RNGScope scope;
  RNG gen;
  
  
  size_t n_train = Z_train.n_rows;
  size_t n_test = Z_test.n_rows;
  size_t p = Z_train.n_cols;
  
  
  
  
  
  double* z_ptr = new double[n_train * p];
  double* z_pred_ptr = new double[n_test * p];
  
  for(size_t i = 0; i < n_train; i++){
    for(size_t j = 0; j < p; j++) z_ptr[j + i*p] = Z_train(i,j);
  }
  for(size_t i = 0; i < n_test; i++){
    for(size_t j = 0; j < p; j++) z_pred_ptr[j + i*p] = Z_test(i,j);
  }
  
  
  
  
  data_info di;
  di.n = n_train;
  di.p = p;
  di.D = D;
  di.z = &z_ptr[0];
  // will update these quantities a bit later
  di.tphi_phi.clear();
  di.phi_beta.clear();
  di.tphi_r.clear();
  
  std::vector<arma::vec> beta_train(n_train, arma::vec<zeros>(D)); // keeps track of every individual's beta
  
  
  for(size_t i = 0; i < n_train; i++){
    Rcpp::NumericVector tmp = Phi_train[i]; // now this holds
  }
  
  
  
  
  
  
  
  
  
  
}
*/
