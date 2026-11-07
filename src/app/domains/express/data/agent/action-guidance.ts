import type { ContextEntry } from "./model-request";

/** Application knowledge about what an action means. It joins the run's
 * context only when a matching action is being answered, never while a view
 * is being created. */
export interface ActionGuidance extends ContextEntry {
  /** The model names events, so match on the actual action name. `origin`
   * is the request the action's view was built for, for generic names such
   * as "review". */
  matches: (actionName: string, origin?: string) => boolean;
}

/** What the shop wants when a checkout form is submitted. The model names the
 * event (review_order, submit_order, ...) and still generates the UI. A bare
 * "review", "submit" or "confirm" counts only on a view built for a checkout,
 * so review_article or a review of a trip form does not match. */
export const CHECKOUT_GUIDANCE: ActionGuidance = {
  description: "Leaf & Light checkout actions",
  matches: (actionName, origin = "") =>
    /order|checkout/i.test(actionName) ||
    (/^(review|submit|confirm)$/i.test(actionName) &&
      /checkout|\bcart\b/i.test(origin)),
  value:
    "The shopper submitted the checkout form to check the order before paying. Answer in the same surface: keep the title, the total, the form and its values, and add a Card titled Review your order with one Text per submitted field showing its actual value as text, plus the order total. The review is only a summary: add no Button or other action, because the shop defines no action after review.",
};
