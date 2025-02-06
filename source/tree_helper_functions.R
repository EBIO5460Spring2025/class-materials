# Find the best split of a dataset
# df:     dataframe with columns x (numeric) and y (numeric)
# n_min:  min number of points in a partition (scalar, integer)
#
best_split <- function(df, n_min) {
    odf <- df[order(df$x),]
    x <- odf$x
    y <- odf$y
    n <- nrow(odf)
    if ( n < 2 * n_min ) stop("We shouldn't get here")
    if ( n == 2 * n_min ) {
        x_split_best <- (x[n_min] + x[n_min+1]) / 2
    } else {
        ssq_best <- Inf
        x_split_best <- NA
        for ( i in (n_min+1):(n-n_min+1) ) {
            x_split <- (x[i-1] + x[i]) / 2
            y_pred_left <- mean(y[x < x_split])
            y_pred_right <- mean(y[x >= x_split])
            y_preds <- ifelse(x < x_split, y_pred_left, y_pred_right)
            ssq <- sum((y - y_preds)^2)
            if ( ssq < ssq_best ) {
                ssq_best <- ssq
                x_split_best <- x_split
            }
        }
    }
    return(x_split_best)
}

# Convert tree structure data frame to row index form
to_row_index_tree <- function(tree) {
    #Initialize data frame
    tree_array <- data.frame(matrix(NA, nrow=max(tree$node), ncol=4))
    names(tree_array) <- names(tree)
    #Convert
    tree_array[tree$node,] <- tree
    return(tree_array)
}

plot_tree <- function(tree, labels=TRUE, splits=TRUE, 
                      digits=getOption("digits")-3, ...) {
    node <- as.integer(tree$node)
    tree_plot(tree_coords(tree), node, ...)
    if ( labels ) {
        add_plot_labels(tree, splits, digits)
    }
}

# From treepl in tree library, pared down
tree_plot <- function(xy, node, ...) {
    x <- xy$x
    y <- xy$y
    parent <- match((node%/%2L), node)
    sibling <- match(ifelse(node%%2L, node - 1L, node + 1L), node)
    xx <- rbind(x, x, x[sibling], x[sibling], NA)
    yy <- rbind(y, y[parent], y[parent], y[sibling], NA)
    plot(range(x), range(y), type = "n", axes = FALSE, xlab = "", ylab = "")
    text(x[1L], y[1L], "|", ...)
    lines(c(xx[, -1L]), c(yy[, -1L]), ...)
}


# From treeco in tree library, pared down
tree_coords <- function(tree) {
    node <- as.integer(tree$node)
    depth <- tree_depth(node)
    x <- -depth
    y <- x
    depth <- -x
    leaves <- tree$type == "leaf"
    x[leaves] <- seq(sum(leaves))
    depth <- split(seq(node)[!leaves], depth[!leaves])
    left_child <- match(node * 2L, node)
    right_child <- match(node * 2 + 1L, node)
    for ( i in rev(depth) ) {
        x[i] <- 0.5 * (x[left_child[i]] + x[right_child[i]])
    }
    list(x = x, y = y)
}

# From tree.depth in tree library
tree_depth <- function(nodes) {
    depth <- floor(log(nodes, base = 2) + 1e-07)
    as.vector(depth - min(depth))
}

# From text.tree in tree library, significantly cut down
add_plot_labels <- function(tree, splits=splits, digits=digits, adj=par("adj"), 
                            xpd=TRUE, ...) {
    oldxpd <- par(xpd = xpd)
    on.exit(par(oldxpd))
    charht <- par("cxy")[2L]
    xy <- tree_coords(tree)
    if (splits) {
        labs <- paste0("<", round(tree$split, 2))
        ind <- !is.na(tree$split)
        text(xy$x[ind], xy$y[ind] + 0.5 * charht, labs[ind], 
            adj = adj, ...)
    }
    leaves <- tree$type == "leaf"
    labs <- format(signif(tree$y_pred[leaves], digits = digits))
    text(xy$x[leaves], xy$y[leaves] - 0.5 * charht, labels = labs, 
            adj = adj, ...)
}



