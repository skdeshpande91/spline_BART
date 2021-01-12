
#include "tree.h"

using std::string;
using std::cout;
using std::endl;

//--------------------------------------------------
// constructors
tree::tree(): mu(arma::zeros<arma::vec>(1)),v(0),c(0),p(0),l(0),r(0) {}

//--------------------------------------------------
//operators
tree& tree::operator=(const tree& rhs)
{
   if(&rhs != this) {
      tonull(); //kill left hand side (this)
      cp(this,&rhs); //copy right hand side to left hand side
   }
   return *this;
}
//--------------------------------------------------
//public functions
// find bottom node pointer given x
//--------------------
tree::tree_cp tree::bn(double *x,xinfo& xi)
{
   if(l==0) return this; //bottom node
   if(x[v] < xi[v][c]) {
      return l->bn(x,xi);
   } else {
      return r->bn(x,xi);
   }
}
//--------------------
//find region for a given variable
void tree::rg(size_t v, int* L, int* U) const
{ 
   if(p==0)  { //no parent
      return;
   }
   if(p->v == v) { //does my parent use v?
      if(this == p->l) { //am I left or right child
         if((int)(p->c) <= (*U)) *U = (p->c)-1; 
      } else {
         if((int)(p->c) >= *L) *L = (p->c)+1; 
      }
   } 
   p->rg(v,L,U);
}
//--------------------
//tree size
size_t tree::treesize() const
{
   if(l==0) return 1;  //if bottom node, tree size is 1
   else return (1+l->treesize()+r->treesize());
}
//--------------------
size_t tree::nnogs() const
{
   if(!l) return 0; //bottom node
   if(l->l || r->l) { //not a nog
      return (l->nnogs() + r->nnogs()); 
   } else { //is a nog
      return 1;
   }
}
size_t tree::nuse(size_t v)
{
   npv nds;
   this->getnodes(nds);
   size_t nu=0; //return value
   for(size_t i=0;i!=nds.size();i++) {
      if(nds[i]->l && nds[i]->v==v) nu+=1;
   }
   return nu;
}

void tree::varsplits(std::set<size_t> &splits, size_t v)
{
   npv nds;
   this->getnodes(nds);
   //size_t nu=0; //return value
   //std::set out;
   for(size_t i=0;i!=nds.size();i++) {
      if(nds[i]->l && nds[i]->v==v) {
        splits.insert(nds[i]->c); //c is index of split rule
      }
   }
}

//--------------------
size_t tree::nbots() const
{
   if(l==0) { //if a bottom node
      return 1;
   } else {
      return l->nbots() + r->nbots();
   }
}


void tree::print(bool pc) const
{
  size_t id = nid();
  size_t pid;
  if(!p) pid = 0; // parent of top node
  else pid = p->nid();
  
  if(pc && (ntype() == 't')) Rcpp::Rcout << "tree size:" << treesize() << std::endl;
  Rcpp::Rcout << " id:" << id;
  //if(n_type() == 'b')
  if(ntype() != 'b') Rcpp::Rcout <<  "(v,c): " << v + 1 << " " << c + 1;
  Rcpp::Rcout << "  type: " << ntype() << " depth: " << depth();
  Rcpp::Rcout << endl;
  
  
  // now recurse down the tree and print each node
  if(pc){
    if(l){
      l->print(pc);
      r->print(pc);
    }
  }
}


//--------------------
//depth of node
size_t tree::depth() const
{
   if(!p) return 0; //no parents
   else return (1+p->depth());
}
//--------------------
// node id
size_t tree::nid() const
//recursion up the tree
{
   if(!p) return 1; //if you don't have a parent, you are the top
   if(this==p->l) return 2*(p->nid()); //if you are a left child
   else return 2*(p->nid())+1; //else you are a right child
}
//--------------------
//node type
char tree::ntype() const
{
   //t:top, b:bottom, n:no grandchildren, i:internal
   if(!p) return 't';
   if(!l) return 'b';
   if(!(l->l) && !(r->l)) return 'n';
   return 'i';
}
//--------------------
//get bottom nodes
//recursion down the tree
void tree::getbots(npv& bv) 
{
   if(l) { //have children
      l->getbots(bv);
      r->getbots(bv);
   } else {
      bv.push_back(this);
   }
}
//--------------------
//get nog nodes
//recursion down the tree
void tree::getnogs(npv& nv) 
{
   if(l) { //have children
      if((l->l) || (r->l)) {  //have grandchildren
         if(l->l) l->getnogs(nv);
         if(r->l) r->getnogs(nv);
      } else {
         nv.push_back(this);
      }
   }
}
//--------------------
//get all nodes
//recursion down the tree
void tree::getnodes(npv& v) 
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
void tree::getnodes(cnpv& v)  const
{
   v.push_back(this);
   if(l) {
      l->getnodes(v);
      r->getnodes(v);
   }
}
//--------------------
//add children to  bot node nid
bool tree::birth(size_t nid,size_t v, size_t c, size_t D)
{
  tree_p np = getptr(nid);
  if(np==0) {
    cout << "error in birth: bottom node not found\n";
    return false; //did not find note with that nid
  }
  if(np->l) {
    cout << "error in birth: found node has children\n";
    return false; //node is not a bottom node
  }

  //add children to bottom node np
  tree_p l = new tree;
  l->mu=arma::zeros<arma::vec>(D);
  tree_p r = new tree;
  r->mu=arma::zeros<arma::vec>(D);
  np->l=l;
  np->r=r;
  np->v = v; np->c=c;
  l->p = np;
  r->p = np;
  return true;
}
//--------------------
//is the node a nog node
bool tree::isnog() const
{
  bool isnog=true;
  if(l) {
    if(l->l || r->l) isnog=false; //one of the children has children.
  } else {
    isnog=false; //no children
  }
  return isnog;
}
//--------------------
//kill children of  nog node nid
bool tree::death(size_t nid, size_t D)
{
  tree_p nb = getptr(nid);
  if(nb==0) {
    cout << "error in death, nid invalid\n";
    return false;
  }
  if(nb->isnog()) {
    delete nb->l;
    delete nb->r;
    nb->l=0;
    nb->r=0;
    nb->v=0;
    nb->c=0;
    nb->mu=arma::zeros<arma::vec>(D);
    return true;
  } else {
    cout << "error in death, node is not a nog node\n";
    return false;
  }
}
//--------------------
//add children to bot node *np
void tree::birthp(tree_p np,size_t v, size_t c, size_t D)
{
  tree_p l = new tree;
  l->mu=arma::zeros<arma::vec>(D);
  tree_p r = new tree;
  r->mu=arma::zeros<arma::vec>(D);
  np->l=l;
  np->r=r;
  np->v = v; np->c=c;
  l->p = np;
  r->p = np;
}
//--------------------
//kill children of  nog node *nb
void tree::deathp(tree_p nb, size_t D)
{
  delete nb->l;
  delete nb->r;
  nb->l=0;
  nb->r=0;
  nb->v=0;
  nb->c=0;
  nb->mu=arma::zeros<arma::vec>(D);
}

//--------------------------------------------------
//private functions
//--------------------
//copy tree o to tree n
void tree::cp(tree_p n, tree_cp o)
//assume n has no children (so we don't have to kill them)
//recursion down
{
  if(n->l) {
    cout << "cp:error node has children\n";
    return;
  }

  n->mu = o->mu;
  n->v = o->v;
  n->c = o->c;

  if(o->l) { //if o has children
    n->l = new tree;
    (n->l)->p = n;
    cp(n->l,o->l);
    n->r = new tree;
    (n->r)->p = n;
    cp(n->r,o->r);
  }
}
//--------------------
//cut back to one node
void tree::tonull()
{
  size_t ts = treesize();
  while(ts>1) { //if false ts=1
    npv nv;
    getnogs(nv);
    for(size_t i=0;i<nv.size();i++) {
        delete nv[i]->l;
        delete nv[i]->r;
        nv[i]->l=0;
        nv[i]->r=0;
    }
    ts = treesize();
  }
  mu=arma::zeros<arma::vec>(1);
  v=0;c=0;
  p=0;l=0;r=0;
}

//--------------------
// get pointer for node from its nid
tree::tree_p tree::getptr(size_t nid)
{
  if(this->nid() == nid) return this; //found it
  if(l==0) return 0; //no children, did not find it
  tree_p lp = l->getptr(nid);
  if(lp) return lp; //found on left
  tree_p rp = r->getptr(nid);
  if(rp) return rp; //found on right
  return 0; //never found it
}
