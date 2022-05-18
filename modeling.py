import xgboost
from sklearn.model_selection import train_test_split
def my_xgboost(X, y, X_predict, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    xgb_model = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                               colsample_bytree=1, max_depth=7)
    xgb_model.fit(X_train,y_train)
    # important variable 생략;
    # xgboost.plot_importance(xgb_model)
    r_sq = xgb_model.score(X_train, y_train)
    #print(r_sq)

    #predictions = xgb_model.predict(X_test)

    predictions = xgb_model.predict(X_predict)
    return predictions