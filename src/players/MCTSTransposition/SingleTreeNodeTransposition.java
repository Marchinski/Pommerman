package players.MCTSTransposition;

import core.GameState;
import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Random;

public class SingleTreeNodeTransposition
{
    public MCTSParamsTransposition params;

    private SingleTreeNodeTransposition parent;
    private SingleTreeNodeTransposition[] children;
    private Random m_rnd;
    private int m_depth;
    private double[] bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
    private int childIdx;
    private int stateIdx;
    private int fmCallsCount;

    private int num_actions;
    private Types.ACTIONS[] actions;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;

    // Transpositions table
    private static Hashtable statesTable = new Hashtable();

    SingleTreeNodeTransposition(MCTSParamsTransposition p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNodeTransposition(MCTSParamsTransposition p, SingleTreeNodeTransposition parent, int childIdx, Random rnd, int num_actions,
                                        Types.ACTIONS[] actions, int fmCallsCount, StateHeuristic sh) {
        this.params = p;
        this.fmCallsCount = fmCallsCount;
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNodeTransposition[num_actions];
        this.childIdx = childIdx;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else
            m_depth = 0;
    }

    // Object that holds information of the stat
    // Will be placed in the hashtable
    class StateInfo{
        int nVisits;
        double actionInfo[][];

        public StateInfo(){
            nVisits = 0;
            actionInfo = new double[6][4];
            for (double[] i : actionInfo) {
                i = new double[]{0,0,Double.MAX_VALUE, -Double.MAX_VALUE};
            }
        }
    }

    void setRootGameState(GameState gs)
    {
        this.rootState = gs;
        if (params.heuristic_method == params.CUSTOM_HEURISTIC)
            this.rootStateHeuristic = new CustomHeuristic(gs);
        else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics
            this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);

        statesTable.clear();
        int rootStateIdx = gs.toString().hashCode();
        statesTable.put(rootStateIdx, new StateInfo());
        this.stateIdx = rootStateIdx;
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        while(!stop){

            GameState state = rootState.copy();
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            // Selection Step
            SingleTreeNodeTransposition selected = treePolicy(state);
            // Simulation Step
            double delta = selected.rollOut(state);
            // Backprobagation Step
            backUp(selected, delta);

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                numIters++;
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                numIters++;
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                fmCallsCount+=params.rollout_depth;
                stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
            }
        }
        //System.out.println(" ITERS " + numIters);
    }

    private SingleTreeNodeTransposition treePolicy(GameState state) {

        SingleTreeNodeTransposition cur = this;

        while (!state.isTerminal() && cur.m_depth < params.rollout_depth)
        {
            // Perform Selection Step
            // Check if current node being explored is a leaf node
            if (cur.notFullyExpanded()) {
                return cur.expand(state);
            }
            // If it is not a leaf node perform UCT to select a child node
            else {
                cur = cur.uct(state);
            }
        }

        return cur;
    }

    // Expand the tree node by selecting the next available action
    private SingleTreeNodeTransposition expand(GameState state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        roll(state, actions[bestAction]);

        int stateID = state.toString().hashCode();
        if (!statesTable.containsKey(stateID))
            statesTable.put(stateID, new StateInfo());

        SingleTreeNodeTransposition tn = new SingleTreeNodeTransposition(params,this,bestAction,this.m_rnd,num_actions,
                actions, fmCallsCount, rootStateHeuristic);
        tn.stateIdx = stateID;
        children[bestAction] = tn;
        return tn;
    }

    // Roll the state forward
    private void roll(GameState gs, Types.ACTIONS act)
    {
        //Simple, all random first, then my position.
        int nPlayers = 4;
        Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];
        int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

        for(int i = 0; i < nPlayers; ++i)
        {
            if(playerId == i)
            {
                actionsAll[i] = act;
            }else {
                int actionIdx = m_rnd.nextInt(gs.nActions());
                actionsAll[i] = Types.ACTIONS.all().get(actionIdx);
            }
        }

        gs.next(actionsAll);

    }

    // Select the best child node using Upper Confidence Bound
    private SingleTreeNodeTransposition uct(GameState state) {
        SingleTreeNodeTransposition selected = null;
        double bestValue = -Double.MAX_VALUE;

        // Get the node state information from the hashtable
        StateInfo info = (StateInfo) statesTable.get(this.stateIdx);
        int i = 0;
        for (double[] action : info.actionInfo)
        {
            double hvVal = action[1];
            double childValue =  hvVal / (action[0] + params.epsilon);

            // Child_Value is the Q value (totValue of node / number of visits to the node)
            childValue = Utils.normalise(childValue, action[2], action[3]);

            // UCT_value is the decision value (Q-value + C*srt(ln(number of visits) / number of times action has been taken from state))
            double uctValue = childValue +
                    params.K * Math.sqrt(Math.log(info.nVisits + 1) / (action[0] + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = this.children[i];
                bestValue = uctValue;
            }

            i++;
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + bounds[0] + " " + bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    //
    private double rollOut(GameState state)
    {
        int thisDepth = this.m_depth;

        while (!finishRollout(state,thisDepth)) {
            int action = safeRandomAction(state);
            roll(state, actions[action]);
            thisDepth++;
        }

        return rootStateHeuristic.evaluateState(state);
    }

    private int safeRandomAction(GameState state)
    {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;

        while(actionsToTry.size() > 0) {

            int nAction = m_rnd.nextInt(actionsToTry.size());
            Types.ACTIONS act = actionsToTry.get(nAction);
            Vector2d dir = act.getDirection().toVec();

            Vector2d pos = state.getPosition();
            int x = pos.x + dir.x;
            int y = pos.y + dir.y;

            if (x >= 0 && x < width && y >= 0 && y < height)
                if(board[y][x] != Types.TILETYPE.FLAMES)
                    return nAction;

            actionsToTry.remove(nAction);
        }

        //Uh oh...
        return m_rnd.nextInt(num_actions);
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean finishRollout(GameState rollerState, int depth)
    {
        if (depth >= params.rollout_depth)      //rollout end condition.
            return true;

        if (rollerState.isTerminal())               //end of game
            return true;

        return false;
    }

    private void backUp(SingleTreeNodeTransposition node, double result)
    {
        SingleTreeNodeTransposition n = node;
        while(n != null)
        {
            StateInfo info = (StateInfo) statesTable.get(n.stateIdx);
            info.nVisits++;

            SingleTreeNodeTransposition np = n.parent;
            if(np != null) {
                info = (StateInfo) statesTable.get(np.stateIdx);
                info.actionInfo[n.childIdx][0]++;
                info.actionInfo[n.childIdx][1] += result;

                if (result < info.actionInfo[n.childIdx][2]) {
                    info.actionInfo[n.childIdx][2] = result;
                }
                if (result > info.actionInfo[n.childIdx][3]) {
                    info.actionInfo[n.childIdx][3] = result;
                }
            }

            n = np;
        }
    }


    int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        StateInfo info = (StateInfo) statesTable.get(stateIdx);
        for (int i=0; i<info.actionInfo.length; i++) {

            if(info.actionInfo[i] != null)
            {
                if(first == -1)
                    first = info.actionInfo[i][0];
                else if(first != info.actionInfo[i][0])
                {
                    allEqual = false;
                }

                double childValue = info.actionInfo[i][0];
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    private int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        StateInfo info = (StateInfo) statesTable.get(stateIdx);
        for (int i=0; i<info.actionInfo.length; i++) {

            if(info.actionInfo[i] != null) {
                double childValue = info.actionInfo[i][1] / (info.actionInfo[i][0] + params.epsilon);
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNodeTransposition tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
