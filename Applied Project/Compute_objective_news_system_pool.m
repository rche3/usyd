function f=Compute_objective_news_system_pool(parameters,Prob)

% Last modified: 05-25-2012

Y_USA=Prob.user.Y_USA;
Y=Prob.user.Y;
X_USA=Prob.user.X_USA;
X=Prob.user.X;
W=Prob.user.W;
T=size(Y,1);
N=size(Y,2)+1;
beta_USA=parameters([1 N+1 N+2]');
u_USA=[0 ; Y_USA(2:end)-X_USA(1:end-1,:)*beta_USA];
U(:,1)=u_USA;
for i=1:N-1;
    Y_i=Y(:,i);
    X_i=X(:,:,i);
    beta_i=parameters([1+i N+1 N+2 N+3 N+4]');
    u_i=Y_i(2:end)-(X_i(1:end-1,:)*beta_i(1:3)+(beta_i(4)*beta_i(5))*u_USA(2:end)+...
        (beta_i(4)*(1-beta_i(5)))*u_USA(1:end-1));
    u_i=[0 ; u_i];
    U(:,i+1)=u_i;
end;
K=size(X_USA,2)+(size(X,2)-1)*(N-1)+(size(X,2)+1)*(N-1)+(N-1);
for t=2:T;
    m_t_USA=[kron([X_USA(t-1,:)],U(t,1))];
    m_t_i=[];
    for i=1:N-1;
        m_t_i=[m_t_i kron(X(t-1,2:end,i),U(t,1)) kron([X(t-1,:,i) U(t-1,1)],U(t,1+i)) U(t,1)*U(t,1+i)];
    end;
    m_t(t,:)=[m_t_USA m_t_i];
end;
m_t=m_t(2:end,:);
m=mean(m_t)';
obj=m'*W*m;
f=obj;
