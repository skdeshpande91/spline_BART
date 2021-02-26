\name{get_rw_precision}
\alias{get_rw_precision}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Get ``precision'' matrix corresponding to random-walk prior
}
\description{
Computes ``precision'' matrix for random-walk prior of orders 1 and 2.
}
\usage{
get_rw_precision(D, rw_order)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{D}{Total number of basis functions.}
  \item{rw_order}{Order of the random walk prior. Currently only supports \code{rw_order} = 1 or 2}
}

\value{A list with entires
\item{K}{D x D precision matrix}
\item{log_det_K}{Log of generalized determinant of K, defined to be product of non-zero eigenvalues.}
\item{rank_K}{Rank of K, which is usually \code{D - rw_order}.}

}